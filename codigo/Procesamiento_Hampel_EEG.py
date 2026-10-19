#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Auditoria y correccion Hampel de potencias EEG de LeftRight y PullPush.

Requiere numpy, pandas y matplotlib. No usa timestamps para analisis ni exportacion.
Las ocurrencias son picos atipicos estadisticos, NO artefactos fisiologicos confirmados.

Ejemplo:
 python Procesamiento_Hampel_EEG.py --entrada /ruta/zip_o_carpeta --salida resultados
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

TEMPORALES = {'timestamp', 'hora', 'fecha'}
PALETA = ['#EF555E', '#ACEF55', '#55EFE5', '#9855EF', '#3B9C43', '#F1999E']


def cargar_csv(payload: bytes):
    """Inferir delimitador y leer manteniendo nombres originales de columnas."""
    cab = payload[:8192].decode('utf-8-sig', errors='replace').splitlines()[0]
    delim = ';' if cab.count(';') > cab.count(',') else ','
    df = pd.read_csv(io.BytesIO(payload), sep=delim, encoding='utf-8-sig', low_memory=False)
    df.columns = [str(x).strip() for x in df.columns]
    if df.columns.duplicated().any():
        raise ValueError('Nombres de columnas repetidos')
    return df, delim


def fuentes(entrada: Path):
    """Encuentra CSV en ZIP LeftRight/PullPush o directorios con esos grupos."""
    if entrada.is_dir():
        zips = sorted(entrada.glob('*.zip'))
        for z in zips:
            g = 'LeftRight' if 'leftright' in z.name.lower() else ('PullPush' if 'pullpush' in z.name.lower() else None)
            if g is None: continue
            with zipfile.ZipFile(z) as arc:
                for name in sorted(arc.namelist()):
                    if name.lower().endswith('.csv') and not name.startswith('__MACOSX/'):
                        yield g, Path(name).name, arc.read(name)
        if not zips:
            for f in sorted(entrada.rglob('*.csv')):
                path = str(f).lower()
                g = 'LeftRight' if 'leftright' in path else ('PullPush' if 'pullpush' in path else None)
                if g: yield g, f.name, f.read_bytes()
    elif entrada.suffix.lower() == '.zip':
        g = 'LeftRight' if 'leftright' in entrada.name.lower() else ('PullPush' if 'pullpush' in entrada.name.lower() else None)
        if g:
            with zipfile.ZipFile(entrada) as arc:
                for name in sorted(arc.namelist()):
                    if name.lower().endswith('.csv'):
                        yield g, Path(name).name, arc.read(name)


def hampel(x: np.ndarray, radio: int = 12, umbral: float = 4.5):
    """Mediana + MAD * 1.4826 en ventanas simetricas exactas (2*radio+1).

    En bordes la ventana se trunca; si MAD=0 no se sustituye para evitar
    falsos positivos por variacion cuantizada o segmentos constantes.
    """
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all(): raise ValueError('La serie debe ser finita')
    xp = np.pad(x, (radio, radio), mode='constant', constant_values=np.nan)
    ventanas = sliding_window_view(xp, 2 * radio + 1)
    with np.errstate(all='ignore'):
        med = np.nanmedian(ventanas, axis=1)
        mad = np.nanmedian(np.abs(ventanas - med[:, None]), axis=1)
    escala = 1.4826 * mad
    mask = (escala > 0) & (np.abs(x - med) > umbral * escala)
    corregida = x.copy()
    corregida[mask] = med[mask]
    return corregida, mask, med, escala


def agrupar_eventos(mask: np.ndarray):
    """Devuelve eventos por segmentos consecutivos marcados, [inicio, fin]."""
    ix = np.flatnonzero(mask)
    if len(ix) == 0: return []
    limites = np.flatnonzero(np.diff(ix) > 1)
    comienzos = np.r_[0, limites + 1]
    finales = np.r_[limites, len(ix) - 1]
    return [(int(ix[i]), int(ix[j])) for i, j in zip(comienzos, finales)]


def analizar_archivo(grupo, nombre, payload, salida, radio=12, umbral=4.5, hz=8.0):
    df, separador = cargar_csv(payload)
    variables = [c for c in df.columns if c.lower() not in TEMPORALES]
    # No se realiza procesamiento temporal: los indices en segundos son solo
    # referencia nominal, nunca horarios reales inferidos del CSV.
    mat = df[variables].apply(pd.to_numeric, errors='coerce')
    validas = np.isfinite(mat.to_numpy(dtype=float)).all(axis=1)
    limpiado = mat.loc[validas].reset_index(drop=True)
    original = limpiado.to_numpy(dtype=float)
    resultado = original.copy()
    masks = np.zeros_like(original, dtype=bool)
    eventos = []
    metricas = []
    for j, col in enumerate(variables):
        x = original[:,j]
        nueva, mask, med, escala = hampel(x, radio=radio, umbral=umbral)
        resultado[:,j] = nueva
        masks[:,j] = mask
        absdif = np.abs(x-nueva)
        mae = float(np.mean(absdif))
        rms = float(np.sqrt(np.mean((x-nueva)**2)))
        num = int(mask.sum())
        metricas.append({
            'Grupo':grupo, 'Archivo':nombre, 'Variable':col,
            'Muestras':len(x), 'Picos detectados':num,
            'Porcentaje corregido':100*num/max(len(x),1),
            'MAE antes-despues':mae,'RMSE antes-despues':rms,
            'Cambio maximo': float(np.max(absdif)),
            'Media original':float(np.mean(x)), 'Media corregida':float(np.mean(nueva)),
            'Desv. original':float(np.std(x)), 'Desv. corregida':float(np.std(nueva)),
        })
        for ini, fin in agrupar_eventos(mask):
            segmento = slice(ini, fin+1)
            de = absdif[segmento]
            scores = np.abs(x[segmento]-med[segmento]) / np.maximum(escala[segmento], 1e-12)
            eventos.append({
                'Grupo': grupo, 'Archivo':nombre, 'Variable':col,
                'Canal':col.split('/')[0] if '/' in col else col,
                'Banda':col.split('/')[-1] if '/' in col else '',
                'Muestra inicio (base 0)': ini, 'Muestra fin (base 0)':fin,
                'Segundo nominal inicio (8 Hz)':round(ini/hz,3),
                'Segundo nominal fin (8 Hz)':round(fin/hz,3),
                'Duracion nominal (s)':round((fin-ini+1)/hz,3),
                'Muestras afectadas':fin-ini+1,
                'Intensidad pico (diferencia absoluta)': float(np.max(de)),
                'Intensidad pico (MAD escalado)': float(np.max(scores)),
            })
    limpio_df = pd.DataFrame(resultado, columns=variables)
    carpeta = salida/'CSV_Limpios'/grupo
    carpeta.mkdir(parents=True,exist_ok=True)
    outname = Path(nombre).stem + '_Hampel_Clean.csv'
    limpio_df.to_csv(carpeta/outname, index=False,sep=separador,float_format='%.10g')
    n = int(masks.sum())
    resumen = {
        'Grupo':grupo,'Archivo':nombre,'Separador CSV':separador,
        'Filas originales':len(df),'Filas incompletas retiradas':int((~validas).sum()),
        'Filas limpias':len(limpiado),'Columnas EEG':len(variables),
        'Muestras EEG evaluadas':int(original.size),'Muestras corregidas':n,
        'Porcentaje EEG corregido': 100*n/max(original.size,1),
        'Eventos consecutivos':len(eventos),
        'Variables con correccion':int(masks.any(axis=0).sum()),
        'MAE global':float(np.mean(np.abs(original-resultado))),
        'RMSE global':float(np.sqrt(np.mean((original-resultado)**2))),
        'Max. correccion':float(np.max(np.abs(original-resultado))),
        'Infinitos posteriores':int(np.isinf(resultado).sum()),
        'Nulos posteriores':int(np.isnan(resultado).sum()),
        'CSV resultado':outname,
    }
    return resumen,metricas,eventos,{'grupo':grupo,'archivo':nombre, 'cols':variables,
                                     'original': original, 'clean':resultado,
                                     'mask':masks}


def graficar_casos(datos, salida, casos=2, hz=8.0):
    """Ilustraciones con muestras y segundos nominales, no marcas reales."""
    salidas=[]
    for grupo in ['LeftRight','PullPush']:
        candidatos=[]
        for d in datos:
            if d['grupo'] != grupo: continue
            # Elegir detectores por impacto relativo, evitando canales constantes
            orig, clean, mask = d['original'],d['clean'],d['mask']
            for j, col in enumerate(d['cols']):
                inds=np.flatnonzero(mask[:,j]);
                if len(inds)==0:continue
                diff=np.abs(orig[:,j]-clean[:,j]); den=max(np.median(np.abs(orig[:,j]-np.median(orig[:,j]))),.01)
                i=int(inds[np.argmax(diff[inds]/den)])
                score=float(diff[i]/den)
                candidatos.append((score,d,j,i))
        candidatos.sort(key=lambda r:r[0],reverse=True)
        escogidos=[]
        for score,d,j,i in candidatos:
            if any(t[1]['archivo']==d['archivo'] for t in escogidos): continue
            escogidos.append((score,d,j,i))
            if len(escogidos)>=casos:break
        for count,(score,d,j,i) in enumerate(escogidos,1):
            orig,clean,mask=d['original'][:,j],d['clean'][:,j],d['mask'][:,j]
            t=np.arange(len(orig))/hz
            a=max(0,i-28); b=min(len(orig),i+29)
            fig,ax=plt.subplots(2,1,figsize=(11,6.6),gridspec_kw={'height_ratios':[1,1.15]})
            ax[0].plot(t,orig,color=PALETA[3],lw=.75,label='Serie original')
            ind=np.flatnonzero(mask)
            ax[0].scatter(t[ind],orig[ind],s=10,c=PALETA[0],label='Picos Hampel',zorder=4)
            ax[0].axvspan(t[a], t[b-1],color=PALETA[1],alpha=.17)
            ax[0].set(xlabel='Tiempo nominal desde el inicio (s; 8 Hz)',ylabel='Potencia por banda (unidad original)',title='Vista completa: '+d['cols'][j]);ax[0].legend(loc='upper right',fontsize=8)
            ax[1].plot(t[a:b],orig[a:b],'-o',ms=2.2,color=PALETA[0],lw=1.3,label='Original')
            ax[1].plot(t[a:b],clean[a:b],'-',lw=1.6,color=PALETA[4],label='Corregida')
            sel=np.flatnonzero(mask[a:b])+a
            ax[1].scatter(t[sel],orig[sel],marker='x',color=PALETA[3],s=30,label='Detecciones')
            ax[1].set(xlabel='Tiempo nominal desde el inicio (s; 8 Hz)',ylabel='Potencia por banda (unidad original)',title=f'Zoom: muestra {i}, {d["cols"][j]}')
            ax[1].legend(loc='upper right',fontsize=8)
            for p in ax: p.grid(alpha=.15)
            fig.suptitle(f'{grupo} • {d["archivo"]} • detección y corrección Hampel',fontsize=12,fontweight='bold')
            fig.tight_layout(rect=(0,0,1,.94))
            img=salida/'Graficas'/f'Caso_{grupo}_{count}.png';img.parent.mkdir(exist_ok=True)
            fig.savefig(img,dpi=160);plt.close(fig)
            salidas.append({'grupo':grupo,'archivo':d['archivo'],'variable':d['cols'][j],
                            'muestra':i,'diferencia':float(abs(orig[i]-clean[i])),
                            'imagen':str(img)})
    return salidas


def resumen_grafico(resumen_df, salida):
    fig,axs=plt.subplots(1,2,figsize=(12,4.3))
    for k,grp in enumerate(['LeftRight','PullPush']):
        tmp=resumen_df[resumen_df['Grupo']==grp]
        etiquetas=tmp['Archivo'].str.replace('.csv','',regex=False).str.split('_').str[0]
        axs[k].bar(range(len(tmp)),tmp['Porcentaje EEG corregido'],color=PALETA[k*3],edgecolor='#222222',linewidth=.4)
        axs[k].set_xticks(range(len(tmp)),etiquetas,rotation=65,ha='right',fontsize=7)
        axs[k].set(title=grp,xlabel='Archivo (identificador)',ylabel='Muestras EEG corregidas (%)')
        axs[k].grid(axis='y',alpha=.2)
    fig.suptitle('Proporción de muestras sustituidas por Hampel, por archivo',weight='bold')
    fig.tight_layout()
    path=salida/'Graficas'/'Comparacion_archivos.png'; fig.savefig(path,dpi=155);plt.close(fig)
    return path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--entrada',default='.',help='Carpeta con ZIP o CSV por grupo')
    ap.add_argument('--salida',default='EEG_Hampel_Salida')
    ap.add_argument('--radio',type=int,default=12,help='Vecinos a cada lado (ventana 25)')
    ap.add_argument('--umbral',type=float,default=4.5,help='Multiplicador de MAD robusta')
    ap.add_argument('--hz',type=float,default=8.0,help='Solo tiempo nominal en reportes')
    args=ap.parse_args()
    if args.radio<1 or args.umbral<=0 or args.hz<=0: ap.error('Parámetros positivos requeridos')
    salida=Path(args.salida);(salida/'Tablas').mkdir(parents=True,exist_ok=True)
    res=[]; vars_=[]; eventos=[];datos=[]
    for g,n,payload in fuentes(Path(args.entrada)):
        print(f'Procesando {g}: {n}',flush=True)
        r,v,e,d=analizar_archivo(g,n,payload,salida,args.radio,args.umbral,args.hz)
        res.append(r);vars_+=v;eventos+=e;datos.append(d)
    if not res: raise FileNotFoundError('No se encontraron CSV de LeftRight/PullPush')
    a=pd.DataFrame(res).sort_values(['Grupo','Archivo'])
    v=pd.DataFrame(vars_).sort_values(['Grupo','Archivo','Variable'])
    e=pd.DataFrame(eventos)
    a.to_csv(salida/'Tablas'/'Resumen_archivos.csv',index=False,encoding='utf-8-sig')
    v.to_csv(salida/'Tablas'/'Analisis_por_variable.csv',index=False,encoding='utf-8-sig')
    e.to_csv(salida/'Tablas'/'Eventos_Hampel.csv',index=False,encoding='utf-8-sig')
    grupos=a.groupby('Grupo',as_index=False).agg({'Filas originales':'sum','Filas incompletas retiradas':'sum','Filas limpias':'sum','Muestras EEG evaluadas':'sum','Muestras corregidas':'sum','Eventos consecutivos':'sum'})
    grupos['Porcentaje EEG corregido']=100*grupos['Muestras corregidas']/grupos['Muestras EEG evaluadas']
    grupos.to_csv(salida/'Tablas'/'Resumen_grupos.csv',index=False,encoding='utf-8-sig')
    imgs=graficar_casos(datos,salida,hz=args.hz)
    resumen_grafico(a,salida)
    (salida/'Tablas'/'Casos_graficados.json').write_text(json.dumps(imgs,indent=2,ensure_ascii=False),encoding='utf-8')
    (salida/'Tablas'/'Parametros.json').write_text(json.dumps({'radio':args.radio,'umbral':args.umbral,'hz_nominal_referencial':args.hz,'nota':'Datos de potencias espectrales, no EEG crudo; timestamps excluidos; el tiempo no es marca absoluta.'},indent=2),encoding='utf-8')
    print('\nResumen por grupo:\n',grupos.to_string(index=False))
    print('\nArchivos:',len(a),' Canales-banda:',len(v),' Eventos:',len(e),flush=True)

if __name__=='__main__': main()
