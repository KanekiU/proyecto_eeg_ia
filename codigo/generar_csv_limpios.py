"""Limpieza de CSV EEG previa a Hampel.
Se eliminan filas incompletas, se excluyen infinitos y se winsorizan extremos
con límites IQR por variable EEG. Se mantienen las columnas temporales.
No se aplica Hampel. Dependencias: pandas, numpy.
"""
from pathlib import Path
import zipfile, io
import numpy as np
import pandas as pd

BASE=Path('/mnt/data')
OUT=BASE/'EEG_Limpieza_Pre_Hampel'
OUT.mkdir(exist_ok=True)
inputs={'LeftRight': BASE/'Grupo_LeftRight(1).zip', 'PullPush': BASE/'Grupo_PullPush(1).zip'}
TEMP={'timestamp','hora','fecha'}
res=[]
for grp, zpath in inputs.items():
    directory=OUT/grp; directory.mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        names=sorted((n for n in z.namelist() if n.lower().endswith('.csv')))
        assert len(names)==10,(grp,len(names))
        for name in names:
            data=z.read(name)
            header=data.splitlines()[0]
            sep=';' if header.count(b';')>header.count(b',') else ','
            df=pd.read_csv(io.BytesIO(data),sep=sep,encoding='utf-8-sig',skipinitialspace=True,low_memory=False)
            df.columns=[str(c).strip() for c in df.columns]
            numeric=[c for c in df.columns if c.lower() not in TEMP]
            # Mantener originales de las variables temporales; convertir únicamente canales EEG
            a=df[numeric].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
            # Una fila incompleta incluye cualquier campo original ausente o cualquier canal no finito.
            valid=(~df.isna().any(axis=1)) & a.notna().all(axis=1)
            d=df.loc[valid].copy().reset_index(drop=True)
            n=a.loc[valid].reset_index(drop=True)
            q1=n.quantile(.25); q3=n.quantile(.75); iqr=q3-q1
            lo=q1-1.5*iqr; hi=q3+1.5*iqr
            below=n.lt(lo,axis=1); above=n.gt(hi,axis=1)
            changed=below|above
            bounded=n.clip(lower=lo,upper=hi,axis=1)
            d.loc[:,numeric]=bounded.to_numpy()
            # writing float with ample precision
            output=directory/(Path(name).stem+'_PreHampel_Clean.csv')
            d.to_csv(output,index=False,sep=sep,float_format='%.15g',encoding='utf-8-sig')
            b=pd.read_csv(output,sep=sep,low_memory=False)
            assert len(b)==len(d) and list(b.columns)==list(d.columns)
            assert np.isfinite(b[numeric].to_numpy(dtype=float)).all()
            assert not b.isna().any().any()
            res.append({'Grupo':grp,'Archivo original':Path(name).name,'Archivo limpio':output.name,
                        'Filas originales':len(df),'Filas incompletas/no finitas eliminadas':int((~valid).sum()),
                        'Filas conservadas':len(d),'Columnas':len(df),'Columnas EEG':len(numeric),
                        'Valores extremos IQR acotados':int(changed.sum().sum()),
                        'Filas con al menos un extremo acotado':int(changed.any(axis=1).sum()),
                        'Delimitador':sep})
for grp in inputs:
    dest=BASE/f'CSV_EEG_{grp}_PreHampel_Clean.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted((OUT/grp).glob('*.csv')): z.write(f,arcname=f.name)
summary=pd.DataFrame(res)
summary.to_csv(OUT/'Resumen_Limpieza_PreHampel.csv',index=False,encoding='utf-8-sig')
with zipfile.ZipFile(BASE/'CSV_EEG_20_PreHampel_Clean.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob('*.csv')): z.write(f,arcname=str(f.relative_to(OUT)))
print(summary.to_string(index=False))
print('\nTOTAL:',summary[['Filas originales','Filas incompletas/no finitas eliminadas','Filas conservadas','Valores extremos IQR acotados']].sum().to_dict())
