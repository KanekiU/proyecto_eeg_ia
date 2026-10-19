#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auditoría reproducible de 20 archivos EEG organizados en dos ZIP independientes.
Ejemplo:
    python EEG_Auditoria_20_CSV.py --entrada . --salida EEG_Auditoria_Resultados
Dependencias: pandas, numpy y matplotlib (más bibliotecas estándar).
NO altera ni corrige los CSV originales.
"""

import argparse
import io
import os
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Permite crear gráficos en equipos sin interfaz gráfica.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRUPOS = ('LeftRight', 'PullPush')
NOMBRES_CLASE = {'class', 'clase', 'label', 'etiqueta', 'target', 'objetivo',
                 'gesture', 'gesto', 'action', 'accion', 'movement', 'movimiento',
                 'event', 'evento', 'marker', 'marcador', 'stimulus', 'estimulo'}


def grupo_de_nombre(nombre):
    canon = re.sub(r'[^a-z]', '', nombre.lower())
    if 'leftright' in canon:
        return 'LeftRight'
    if 'pullpush' in canon or 'pushpull' in canon:
        return 'PullPush'
    return None


def descubrir_csv(entrada):
    """Admite ZIP o carpetas; produce bytes, tamaño y nombre real sin alterar el origen."""
    hallados = {grupo: [] for grupo in GRUPOS}
    for zpath in sorted(entrada.rglob('*.zip')):
        grupo_zip = grupo_de_nombre(zpath.name)
        if not grupo_zip:
            continue
        with zipfile.ZipFile(zpath) as zf:
            for info in sorted(zf.infolist(), key=lambda a: a.filename.lower()):
                if info.is_dir() or not info.filename.lower().endswith('.csv'):
                    continue
                nombre = Path(info.filename).name
                grupo = grupo_de_nombre(nombre) or grupo_zip
                hallados[grupo].append((nombre, zf.read(info), info.file_size, zpath.name))
    grupos_en_zips = {g for g in GRUPOS if hallados[g]}
    for path in sorted(entrada.rglob('*.csv')):
        grupo = grupo_de_nombre(path.name) or grupo_de_nombre(str(path.parent))
        if grupo and grupo not in grupos_en_zips:
            hallados[grupo].append((path.name, path.read_bytes(), path.stat().st_size,
                                    str(path.relative_to(entrada))))
    return hallados


def leer_csv(datos):
    """Detecta separador y codificación. Limpia únicamente espacios del encabezado."""
    primera = next((linea for linea in datos.splitlines() if linea.strip()), b'')
    sep = ';' if primera.count(b';') > primera.count(b',') else ','
    errores = []
    for encoding in ('utf-8-sig', 'cp1252'):
        try:
            df = pd.read_csv(io.BytesIO(datos), sep=sep, encoding=encoding,
                             skipinitialspace=True, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            if df.columns.duplicated().any():
                raise ValueError('Nombres de columnas duplicados tras quitar espacios')
            return df, sep, encoding
        except (UnicodeError, pd.errors.ParserError) as exc:
            errores.append(str(exc))
    raise ValueError('No se pudo leer el CSV: ' + '; '.join(errores))


def es_temporal(nombre):
    return bool(re.search(r'timestamp|fecha|date|hora|time', str(nombre), re.I))


def columna_clase(nombre):
    token = re.sub(r'[^a-z]', '', nombre.lower())
    return token in NOMBRES_CLASE or token.startswith(('classid', 'labelid', 'targetid'))


def columna_indice(nombre, serie, n):
    if re.fullmatch(r'(?i)(unnamed:\s*\d+|index|indice|idx|rowid|sampleid)', nombre):
        return True
    if not pd.api.types.is_numeric_dtype(serie) or len(serie) != n or n == 0:
        return False
    numeros = pd.to_numeric(serie, errors='coerce')
    if numeros.isna().any() or not np.isfinite(numeros.to_numpy(dtype=float)).all():
        return False
    inicio = numeros.iloc[0]
    return inicio in (0, 1) and bool(np.allclose(numeros.to_numpy(),
                                                  np.arange(n) + inicio, rtol=0, atol=0))


def analizar_tiempo(df, temporal):
    """Audita la precisión temporal sin inventar subsegundos ni corregir marcas."""
    resultados = []
    for col in temporal:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            numerica = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
            diff = numerica.diff()
            fechas = pd.to_datetime(numerica, unit='s', utc=True, errors='coerce')
            resultados.append({'Variable': col, 'Interpretación': 'Unix (segundos) si es plausible',
                               'Valores no interpretables': int(fechas.isna().sum()),
                               'Marcas repetidas consecutivas': int((diff == 0).sum()),
                               'Retrocesos': int((diff < 0).sum()),
                               'Intervalo mediano (s)': float(diff[diff > 0].median())
                               if (diff > 0).any() else np.nan,
                               'Observación': ('Resolución temporal insuficiente para diferenciar muestras'
                                               if (diff == 0).any() else 'Sin repeticiones consecutivas')})
        else:
            # Hora aislada y fechas locales no permiten inferir una zona horaria.
            txt = s.astype('string')
            resultados.append({'Variable': col, 'Interpretación': 'Texto temporal (sin huso supuesto)',
                               'Valores no interpretables': int(s.isna().sum()),
                               'Marcas repetidas consecutivas': int((txt == txt.shift()).fillna(False).sum()),
                               'Retrocesos': np.nan, 'Intervalo mediano (s)': np.nan,
                               'Observación': 'Comprobar formato y zona horaria antes de combinar'})
    return resultados


def medir_variable(serie, temporal=False, indice=False, clase=False):
    """Calcula faltantes, infinitos, atípicos IQR y saltos sin confundir metadatos con EEG."""
    n = len(serie)
    nulos = int(serie.isna().sum())
    # NaN numérico es un subconjunto de nulos: NO se suman ambos totales.
    nan = int(serie.map(lambda x: isinstance(x, (float, np.floating)) and np.isnan(x)).sum())
    unicos = int(serie.nunique(dropna=True))
    es_num = pd.api.types.is_numeric_dtype(serie)
    infinitos = atipicos = saltos = 0
    minimo = maximo = magnitud = np.nan
    pct_atip = np.nan
    umbral_salto = np.nan
    if es_num:
        arr = pd.to_numeric(serie, errors='coerce').to_numpy(dtype=float)
        infinitos = int(np.isinf(arr).sum())
        limpio = arr[np.isfinite(arr)]
        if limpio.size:
            minimo, maximo = float(limpio.min()), float(limpio.max())
        # Excluir timestamps, índices y objetivos: el IQR debe aplicarse a señales.
        if not temporal and not indice and not clase and limpio.size:
            q1, q3 = np.quantile(limpio, [0.25, 0.75])
            iqr = q3 - q1
            atipicos = int(((limpio < q1 - 1.5 * iqr) |
                            (limpio > q3 + 1.5 * iqr)).sum())
            pct_atip = round(100 * atipicos / limpio.size, 5)
            if len(arr) > 1:
                # Saltos entre muestras adyacentes, excluyendo pares con NaN/inf.
                delta = np.diff(arr)
                abs_delta = np.abs(delta[np.isfinite(delta)])
                if abs_delta.size:
                    d1, d3 = np.quantile(abs_delta, [0.25, 0.75])
                    umbral_salto = float(d3 + 3 * (d3 - d1))
                    saltos = int((abs_delta > umbral_salto).sum())
                    magnitud = float(abs_delta.max())
    constante = unicos <= 1  # Incluye columnas totalmente vacías.
    return {'Tipo de dato': str(serie.dtype), 'Valores únicos': unicos,
            'Nulos': nulos, '% nulos': round(100 * nulos / n, 5) if n else 0.0,
            'NaN': nan, 'Infinitos': infinitos, 'Atípicos': atipicos,
            '% atípicos': pct_atip, 'Constante': bool(constante),
            'Cambios abruptos': saltos, 'Máx. cambio': magnitud,
            'Umbral cambio': umbral_salto, 'Mínimo finito': minimo,
            'Máximo finito': maximo, 'Temporal': temporal,
            'Índice potencial': indice, 'Clase candidata': clase}


def analizar_archivo(nombre, datos, tamano, origen, grupo, destino):
    df, separador, encoding = leer_csv(datos)
    columnas_tiempo = [c for c in df if es_temporal(c)]
    clase = [c for c in df if columna_clase(c)]
    indices = [c for c in df if columna_indice(c, df[c], len(df)) and c not in columnas_tiempo]
    numericas = list(df.select_dtypes(include='number').columns)
    categoricas = [c for c in df.select_dtypes(include=['object', 'category', 'string']).columns
                   if c not in columnas_tiempo]
    n = len(df)
    # Sólo ordenar temporalmente para examinar diferencias consecutivas.
    ordenada = df
    tiempo_numerico = next((c for c in columnas_tiempo
                            if pd.api.types.is_numeric_dtype(df[c])), None)
    if tiempo_numerico:
        orden = pd.to_numeric(df[tiempo_numerico], errors='coerce')
        ordenada = df.assign(__orden=orden).sort_values('__orden', kind='stable',
                                                        na_position='last').drop(columns='__orden')
    detalles = []
    for variable in df.columns:
        datos_variable = ordenada[variable] if variable in numericas else df[variable]
        metricas = medir_variable(datos_variable, variable in columnas_tiempo,
                                  variable in indices, variable in clase)
        detalles.append({'Variable': variable, **metricas})
    tabla = pd.DataFrame(detalles)
    incompletos = df.isna().sum(axis=1)
    n_incompletos = int((incompletos > 0).sum())
    top = (pd.DataFrame({'Fila CSV (encabezado=1)': incompletos.index + 2,
                         'Campos faltantes': incompletos.to_numpy()})
           .query('`Campos faltantes` > 0')
           .sort_values('Campos faltantes', ascending=False).head(15))
    base = Path(nombre).stem
    ruta = destino / grupo / base
    ruta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta / 'Tabla_3_variables.csv', index=False, encoding='utf-8-sig')
    top.to_csv(ruta / 'Registros_incompletos_top15.csv', index=False, encoding='utf-8-sig')
    cats = []
    for c in categoricas:
        unicos = int(df[c].nunique(dropna=True))
        ejemplos = [str(x) for x in df[c].dropna().unique()[:15]]
        cats.append({'Variable': c, 'Valores únicos': unicos,
                     'Valores/ejemplos (máximo 15)': ' | '.join(ejemplos),
                     'Lista completa': unicos <= 15})
    pd.DataFrame(cats, columns=['Variable', 'Valores únicos',
                                'Valores/ejemplos (máximo 15)', 'Lista completa']).to_csv(
                                    ruta / 'Categorias.csv', index=False, encoding='utf-8-sig')
    tiempos = pd.DataFrame(analizar_tiempo(df, columnas_tiempo))
    tiempos.to_csv(ruta / 'Timestamps.csv', index=False, encoding='utf-8-sig')
    # La suma de IQR/saltos cuenta ocurrencias POR VARIABLE, no filas distintas.
    resumen_estructura = {
        'Grupo': grupo, 'Archivo': nombre, 'Extensión': Path(nombre).suffix,
        'Filas': n, 'Columnas': len(df.columns), 'Registros': n,
        'Tamaño (KB)': round(tamano / 1024, 2),
        'Variables numéricas': len(numericas), 'Variables categóricas': len(categoricas),
        'Timestamps': ' | '.join(columnas_tiempo) if columnas_tiempo else 'No',
        'Clase identificada': ' | '.join(clase) if clase else 'No identificada',
        'Índice potencial': ' | '.join(indices) if indices else 'No identificado',
        'Separador': separador, 'Codificación': encoding, 'Origen': origen,
        'Columnas exactas': ' | '.join(df.columns),
    }
    resumen_calidad = {
        'Grupo': grupo, 'Archivo': nombre,
        'Valores nulos': int(tabla['Nulos'].sum()),
        'NaN': int(tabla['NaN'].sum()),
        'Infinitos': int(tabla['Infinitos'].sum()),
        'Duplicados': int(df.duplicated().sum()),
        '% duplicados': round(100 * df.duplicated().sum() / n, 5) if n else 0.,
        'Columnas constantes': int(tabla['Constante'].sum()),
        'Valores atípicos': int(tabla['Atípicos'].sum()),
        'Cambios abruptos': int(tabla['Cambios abruptos'].sum()),
        'Registros incompletos': n_incompletos,
        '% registros incompletos': round(100 * n_incompletos / n, 5) if n else 0.,
        'Máximo faltantes por fila': int(incompletos.max()) if n else 0,
        'Variables con atípicos': int((tabla['Atípicos'] > 0).sum()),
        'Variables con saltos': int((tabla['Cambios abruptos'] > 0).sum()),
    }
    print(f"  {nombre}: {n} filas × {len(df.columns)} columnas; "
          f"nulos={resumen_calidad['Valores nulos']}; "
          f"atípicos={resumen_calidad['Valores atípicos']}; "
          f"saltos={resumen_calidad['Cambios abruptos']}")
    return resumen_estructura, resumen_calidad, tiempos.assign(Archivo=nombre, Grupo=grupo)


def graficar_grupo(calidad, grupo, destino):
    if calidad.empty:
        return
    for metrica, nombre_png in [('Valores nulos', 'Valores_nulos.png'),
                                ('Valores atípicos', 'Valores_atipicos.png'),
                                ('Cambios abruptos', 'Cambios_abruptos.png')]:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(calidad)), calidad[metrica].to_numpy(dtype=float), color='#246A73')
        ax.set_xticks(range(len(calidad)), [Path(x).stem for x in calidad['Archivo']],
                      rotation=55, ha='right', fontsize=8)
        ax.set_title(f'{grupo} — {metrica} por archivo')
        ax.set_ylabel('Ocurrencias (suma por variable)')
        ax.grid(axis='y', alpha=.25)
        fig.tight_layout()
        fig.savefig(destino / grupo / nombre_png, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Auditoría de datos EEG por dos grupos')
    parser.add_argument('--entrada', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--salida', type=Path,
                        default=Path(__file__).resolve().parent / 'EEG_Auditoria_Resultados')
    args = parser.parse_args()
    entrada, destino = args.entrada.resolve(), args.salida.resolve()
    destino.mkdir(parents=True, exist_ok=True)
    elementos = descubrir_csv(entrada)
    estructuras, calidades, tiempos_globales, errores = [], [], [], []
    for grupo in GRUPOS:
        (destino / grupo).mkdir(parents=True, exist_ok=True)
        print(f'\n==== GRUPO {grupo.upper()} — {len(elementos[grupo])} CSV ====')
        if len(elementos[grupo]) != 10:
            print(f'ATENCIÓN: se esperaban 10 archivos y se encontraron {len(elementos[grupo])}')
        for nombre, datos, tamano, origen in elementos[grupo]:
            try:
                est, cal, tiempos = analizar_archivo(nombre, datos, tamano, origen,
                                                    grupo, destino)
                estructuras.append(est)
                calidades.append(cal)
                tiempos_globales.append(tiempos)
            except Exception as exc:
                mensaje = {'Grupo': grupo, 'Archivo': nombre, 'Error': str(exc)}
                errores.append(mensaje)
                print(f'ERROR {nombre}: {exc}')
    estructura_df = pd.DataFrame(estructuras)
    calidad_df = pd.DataFrame(calidades)
    for grupo in GRUPOS:
        sube = estructura_df[estructura_df['Grupo'] == grupo].copy()
        subc = calidad_df[calidad_df['Grupo'] == grupo].copy()
        sube.to_csv(destino / grupo / 'Tabla_1_Estructura.csv', index=False, encoding='utf-8-sig')
        subc.to_csv(destino / grupo / 'Tabla_2_Calidad.csv', index=False, encoding='utf-8-sig')
        graficar_grupo(subc, grupo, destino)
        print('\nTABLA 1 - ESTRUCTURA:', grupo)
        print(sube.drop(columns=['Columnas exactas', 'Origen', 'Codificación']).to_string(index=False))
        print('\nTABLA 2 - CALIDAD:', grupo)
        print(subc.to_string(index=False))
    if tiempos_globales:
        pd.concat(tiempos_globales, ignore_index=True).to_csv(
            destino / 'Auditoria_timestamps.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(errores, columns=['Grupo', 'Archivo', 'Error']).to_csv(
        destino / 'Errores_lectura.csv', index=False, encoding='utf-8-sig')
    resumen = []
    for grupo in GRUPOS:
        est = estructura_df[estructura_df['Grupo'] == grupo]
        cal = calidad_df[calidad_df['Grupo'] == grupo]
        resumen.append({'Grupo': grupo, 'Archivos': len(est), 'Filas': int(est['Filas'].sum()),
                        'Valores nulos': int(cal['Valores nulos'].sum()),
                        'NaN': int(cal['NaN'].sum()), 'Infinitos': int(cal['Infinitos'].sum()),
                        'Duplicados': int(cal['Duplicados'].sum()),
                        'Constantes (archivo-variable)': int(cal['Columnas constantes'].sum()),
                        'Atípicos (ocurrencias)': int(cal['Valores atípicos'].sum()),
                        'Saltos (ocurrencias)': int(cal['Cambios abruptos'].sum()),
                        'Registros incompletos': int(cal['Registros incompletos'].sum())})
    r = pd.DataFrame(resumen)
    r.to_csv(destino / 'Comparacion_grupos_solo_metricas.csv', index=False, encoding='utf-8-sig')
    print('\n==== COMPARACIÓN GLOBAL (SIN MEZCLAR REGISTROS) ====')
    print(r.to_string(index=False))
    print(f'\nArchivos con error: {len(errores)}; resultados guardados en: {destino}')


if __name__ == '__main__':
    main()
