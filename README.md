# HidroSed · Módulo Doble Cuenca v1

Primera versión independiente para probar la nueva lógica de **doble punto de control** antes de incorporarla a HidroSed global.

## Objetivo

El módulo permite trabajar con dos cuencas calculadas sobre un mismo DEM:

1. **PC-HIDRO**: punto de control hidrológico. Define la subcuenca usada para hidrología, caudal base, morfometría y tiempo de concentración.
2. **PC-DESCARGA**: punto de descarga o cierre de cuenca de soporte. Define la cuenca amplia que contiene el tramo hidráulico, respaldo topográfico, corredor del cauce y curvas de nivel.

Luego calcula:

```text
Intercuenca = Cuenca PC-DESCARGA - Cuenca PC-HIDRO
```

y permite definir un caudal adicional al cauce.

## Funciones v1

- Carga de DEM GeoTIFF común.
- Descarga opcional de DEM desde OpenTopography con API Key.
- Reproyección automática a UTM si el DEM viene en coordenadas geográficas.
- Delimitación de cuenca mediante Priority-Flood + D8 + acumulación de flujo.
- Ajuste automático de cada punto al cauce de mayor acumulación dentro de un radio configurable.
- Cálculo de dos cuencas: hidrológica y descarga/soporte.
- Cálculo de área incremental simple y geométrica.
- Validación de contención entre cuencas.
- Caudal adicional opcional:
  - sin aporte adicional;
  - manual;
  - estimado por proporción de área incremental.
- Curvas de nivel opcionales desde DEM dentro de la cuenca de descarga.
- Exportación de resultados en ZIP, KMZ, GeoJSON, Excel y CSV.

## Instalación local

```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
# .venv\Scripts\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Insumos recomendados

### DEM

- GeoTIFF proyectado en UTM WGS84, idealmente EPSG:32719 para Coquimbo.
- También puede usarse DEM geográfico; la aplicación lo reproyecta automáticamente.
- Para pruebas rápidas usar DEM recortado.

### Puntos

- KMZ/KML con geometría Point para PC-HIDRO.
- KMZ/KML con geometría Point para PC-DESCARGA.
- Alternativamente, ingreso manual UTM WGS84.

## Salidas

El ZIP de resultados incluye:

- `01_doble_cuenca_intercuenca.kmz`
- `02_doble_cuenca_intercuenca.geojson`
- `03_metricas.csv`
- `04_caudales.csv`
- `05_resultados.xlsx`
- `06_resumen_integracion_hidrosed.json`
- `07_dem_procesado_utm.tif` si se activa la opción de incluir DEM.

## Advertencias técnicas

- El DEM global de 30 m sirve para cuencas, morfometría y respaldo cartográfico, pero no reemplaza topografía de detalle para secciones hidráulicas.
- Si la cuenca hidrológica no queda dentro de la cuenca de descarga, debe revisarse el orden de los puntos, el radio de ajuste al cauce o la calidad del DEM.
- Las curvas de nivel generadas desde DEM son de apoyo; no son curvas levantadas en terreno.

## Pendiente para v2

- Carga de eje hidráulico.
- Generación de riberas izquierda y derecha.
- Carga de curvas KMZ de respaldo topográfico.
- Carga de secciones Excel tipo HEC-RAS.
- Secciones trapezoidales y rectangulares editables.
- Exportación de secciones para HidroSed global.
