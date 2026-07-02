"""
HidroSed - Módulo Doble Cuenca v1
----------------------------------

Primera versión independiente para trabajar con dos puntos de control:

1) PC-HIDRO: punto de control hidrológico para subcuenca de cálculo.
2) PC-DESCARGA: punto de descarga/cierre de cuenca topográfica de soporte.

Funciones v1:
- Cargar DEM GeoTIFF o descargar DEM global desde OpenTopography.
- Reproyectar automáticamente a UTM si el DEM viene en coordenadas geográficas.
- Delimitar dos cuencas usando Priority-Flood + D8 + acumulación.
- Calcular área incremental: cuenca descarga - cuenca hidrológica.
- Validar contención geométrica entre cuencas.
- Permitir caudal adicional opcional al cauce, manual o estimado por área incremental.
- Generar curvas de nivel desde DEM dentro de la cuenca de descarga.
- Exportar KMZ/KML, GeoJSON, CSV, Excel y resumen JSON.

Nota: Las curvas generadas desde DEM global 30 m son de apoyo/cartográficas, no reemplazan topografía de terreno.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile
import zipfile
import heapq
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

try:
    import requests
except Exception:
    requests = None

try:
    import rasterio
    from rasterio.features import shapes
    from rasterio.transform import Affine
    from rasterio.crs import CRS
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.io import MemoryFile
except Exception as exc:  # pragma: no cover
    st.error("No se pudo importar rasterio. Instala las dependencias de requirements.txt.")
    raise exc

try:
    from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon, LineString, MultiLineString, box
    from shapely.ops import unary_union, transform as shp_transform
except Exception as exc:  # pragma: no cover
    st.error("No se pudo importar shapely. Instala las dependencias de requirements.txt.")
    raise exc

try:
    from pyproj import Transformer, CRS as PyCRS
except Exception as exc:  # pragma: no cover
    st.error("No se pudo importar pyproj. Instala las dependencias de requirements.txt.")
    raise exc

try:
    from skimage import measure
except Exception:
    measure = None

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------

APP_TITLE = "HidroSed · Módulo Doble Cuenca v1"
OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"
SUPPORTED_DEMS = ["COP30", "NASADEM", "SRTMGL1", "SRTMGL3", "COP90", "SRTMGL3"]
DEFAULT_DEM = "COP30"

st.set_page_config(page_title=APP_TITLE, page_icon="🌊", layout="wide")

D8_OFFSETS: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ControlPoint:
    name: str
    x_dem: float
    y_dem: float
    lon: Optional[float]
    lat: Optional[float]
    source: str


@dataclass
class WatershedResult:
    name: str
    dem: np.ndarray
    filled_dem: np.ndarray
    valid: np.ndarray
    transform: Affine
    crs: CRS
    cell_area_m2: float
    res_x: float
    res_y: float
    outlet_rc: Tuple[int, int]
    snapped_rc: Tuple[int, int]
    flow_to: np.ndarray
    accumulation: np.ndarray
    basin_mask: np.ndarray
    stream_mask: np.ndarray
    basin_geom: object
    metrics: Dict[str, Any]

# -----------------------------------------------------------------------------
# Utilidades KML/KMZ
# -----------------------------------------------------------------------------

def _extract_kml_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".kmz"):
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError("El KMZ no contiene un archivo .kml interno.")
            preferred = next((n for n in kml_names if n.lower().endswith("doc.kml")), kml_names[0])
            return zf.read(preferred).decode("utf-8", errors="ignore")
    if name.endswith(".kml"):
        return raw.decode("utf-8", errors="ignore")
    raise ValueError("El archivo debe ser .kml o .kmz.")


def parse_first_kml_point(uploaded_file) -> Tuple[float, float, Optional[float]]:
    """Extrae el primer Point de un KML/KMZ. KML/KMZ almacena lon,lat,z."""
    kml_text = _extract_kml_from_upload(uploaded_file)
    root = ET.fromstring(kml_text.encode("utf-8"))

    # Prioridad: coordenadas dentro de <Point>
    for elem in root.iter():
        if elem.tag.lower().endswith("point"):
            for child in elem.iter():
                if child.tag.lower().endswith("coordinates") and child.text:
                    coords = child.text.strip().replace("\n", " ").replace("\t", " ").split()
                    if coords:
                        parts = [p for p in coords[0].split(",") if p != ""]
                        if len(parts) >= 2:
                            lon = float(parts[0])
                            lat = float(parts[1])
                            z = float(parts[2]) if len(parts) >= 3 else None
                            return lon, lat, z

    # Respaldo: primer coordinates genérico.
    coords_nodes = []
    for elem in root.iter():
        if elem.tag.endswith("coordinates") and elem.text:
            coords_nodes.append(elem.text.strip())
    if not coords_nodes:
        match = re.search(r"<coordinates[^>]*>(.*?)</coordinates>", kml_text, re.S | re.I)
        if not match:
            raise ValueError("No se encontraron coordenadas en el KML/KMZ.")
        coords_text = match.group(1).strip()
    else:
        coords_text = coords_nodes[0]

    first = coords_text.replace("\n", " ").replace("\t", " ").split()[0]
    parts = [p for p in first.split(",") if p != ""]
    if len(parts) < 2:
        raise ValueError("El nodo coordinates no tiene formato lon,lat[,z].")
    lon = float(parts[0])
    lat = float(parts[1])
    z = float(parts[2]) if len(parts) >= 3 else None
    return lon, lat, z


def kml_escape(text: Any) -> str:
    return escape(str(text), quote=True)


def coords_to_kml(coords: Iterable[Tuple[float, float]]) -> str:
    return " ".join([f"{x:.8f},{y:.8f},0" for x, y in coords])


def polygon_to_kml_placemarks(geom_ll, name: str, style_id: str, description: str = "") -> str:
    if geom_ll is None or geom_ll.is_empty:
        return ""
    if isinstance(geom_ll, Polygon):
        polygons = [geom_ll]
    elif isinstance(geom_ll, MultiPolygon):
        polygons = list(geom_ll.geoms)
    else:
        return ""
    out = []
    for i, poly in enumerate(polygons, start=1):
        if poly.is_empty:
            continue
        outer = coords_to_kml(poly.exterior.coords)
        out.append(f"""
        <Placemark>
          <name>{kml_escape(name)} {i}</name>
          <description>{kml_escape(description)}</description>
          <styleUrl>#{style_id}</styleUrl>
          <Polygon><outerBoundaryIs><LinearRing><coordinates>{outer}</coordinates></LinearRing></outerBoundaryIs></Polygon>
        </Placemark>""")
    return "\n".join(out)


def point_to_kml(lon: float, lat: float, name: str, style_id: str, description: str = "") -> str:
    return f"""
        <Placemark>
          <name>{kml_escape(name)}</name>
          <description>{kml_escape(description)}</description>
          <styleUrl>#{style_id}</styleUrl>
          <Point><coordinates>{lon:.8f},{lat:.8f},0</coordinates></Point>
        </Placemark>"""


def line_to_kml(line_ll: LineString, name: str, style_id: str, description: str = "") -> str:
    coords = coords_to_kml(line_ll.coords)
    return f"""
        <Placemark>
          <name>{kml_escape(name)}</name>
          <description>{kml_escape(description)}</description>
          <styleUrl>#{style_id}</styleUrl>
          <LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString>
        </Placemark>"""


def make_kmz_from_kml(kml_bytes: bytes, internal_name: str = "doc.kml") -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(internal_name, kml_bytes)
    return bio.getvalue()

# -----------------------------------------------------------------------------
# CRS y DEM
# -----------------------------------------------------------------------------

def utm_epsg(zone: int, hemisphere: str) -> str:
    if not 1 <= int(zone) <= 60:
        raise ValueError("El huso UTM debe estar entre 1 y 60.")
    hemi = hemisphere.upper().strip()
    code = 32600 + int(zone) if hemi.startswith("N") else 32700 + int(zone)
    return f"EPSG:{code}"


def estimate_utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def is_projected_metre_crs(crs: CRS) -> bool:
    try:
        pycrs = PyCRS.from_user_input(crs)
        axis_units = [ax.unit_name.lower() for ax in pycrs.axis_info]
        return pycrs.is_projected and any("metre" in u or "meter" in u for u in axis_units)
    except Exception:
        return bool(crs and crs.is_projected)


def transform_point_xy(x: float, y: float, src_crs: str | CRS, dst_crs: CRS) -> Tuple[float, float]:
    transformer = Transformer.from_crs(PyCRS.from_user_input(src_crs), PyCRS.from_user_input(dst_crs), always_xy=True)
    return transformer.transform(x, y)


def rc_from_xy(transform: Affine, x: float, y: float) -> Tuple[int, int]:
    col_f, row_f = ~transform * (x, y)
    return int(math.floor(row_f)), int(math.floor(col_f))


def xy_from_rc(transform: Affine, row: int, col: int) -> Tuple[float, float]:
    x, y = transform * (col + 0.5, row + 0.5)
    return float(x), float(y)


def normalize_dem_array(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    arr = arr.astype("float64")
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def read_dem_from_upload(uploaded_file) -> Tuple[np.ndarray, Affine, CRS, Optional[float], Dict[str, Any]]:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".tif"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        with rasterio.open(tmp_path) as ds:
            arr = normalize_dem_array(ds.read(1), ds.nodata)
            transform = ds.transform
            crs = ds.crs
            nodata = ds.nodata
            bounds = ds.bounds
            meta = {
                "source_name": uploaded_file.name,
                "width": ds.width,
                "height": ds.height,
                "crs": crs.to_string() if crs else None,
                "bounds": tuple(bounds),
            }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    if crs is None:
        raise ValueError("El DEM no tiene CRS definido. Debe estar georreferenciado.")
    return arr, transform, crs, nodata, meta


def reproject_dem_to_projected(
    dem: np.ndarray,
    transform: Affine,
    crs: CRS,
    dst_crs: Optional[CRS] = None,
    nodata_value: float = -9999.0,
) -> Tuple[np.ndarray, Affine, CRS, Dict[str, Any]]:
    """Reproyecta DEM a CRS proyectado en metros si viene geográfico."""
    if is_projected_metre_crs(crs):
        return dem, transform, crs, {"reprojected": False, "dst_crs": crs.to_string()}

    # Estimar UTM desde centro del raster en lon/lat.
    nrows, ncols = dem.shape
    cx, cy = transform * (ncols / 2, nrows / 2)
    if dst_crs is None:
        dst_crs = estimate_utm_crs_from_lonlat(float(cx), float(cy))

    left, top = transform * (0, 0)
    right, bottom = transform * (ncols, nrows)
    west, east = sorted([left, right])
    south, north = sorted([bottom, top])

    dst_transform, width, height = calculate_default_transform(
        crs, dst_crs, ncols, nrows, west, south, east, north
    )
    src = np.where(np.isfinite(dem), dem, nodata_value).astype("float32")
    dst = np.full((height, width), nodata_value, dtype="float32")
    reproject(
        source=src,
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        src_nodata=nodata_value,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=nodata_value,
        resampling=Resampling.bilinear,
    )
    dst_arr = normalize_dem_array(dst.astype("float64"), nodata_value)
    return dst_arr, dst_transform, dst_crs, {
        "reprojected": True,
        "src_crs": crs.to_string(),
        "dst_crs": dst_crs.to_string(),
        "width": width,
        "height": height,
    }


def write_dem_to_geotiff_bytes(dem: np.ndarray, transform: Affine, crs: CRS) -> bytes:
    nodata_value = -9999.0
    arr = np.where(np.isfinite(dem), dem, nodata_value).astype("float32")
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value,
        "compress": "lzw",
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(arr, 1)
        return memfile.read()

# -----------------------------------------------------------------------------
# OpenTopography
# -----------------------------------------------------------------------------

def km_per_degree_lon(lat: float) -> float:
    return max(1e-6, 111.320 * math.cos(math.radians(lat)))


def bbox_area_km2(south: float, north: float, west: float, east: float) -> float:
    midlat = (south + north) / 2
    return abs((north - south) * 111.320) * abs((east - west) * km_per_degree_lon(midlat))


def bbox_from_points(points_lonlat: List[Tuple[float, float]], buffer_km: float) -> Tuple[float, float, float, float]:
    if not points_lonlat:
        raise ValueError("No hay puntos lon/lat para calcular bbox.")
    lons = [p[0] for p in points_lonlat]
    lats = [p[1] for p in points_lonlat]
    lat_mid = (min(lats) + max(lats)) / 2
    dlat = buffer_km / 111.320
    dlon = buffer_km / km_per_degree_lon(lat_mid)
    return min(lats) - dlat, max(lats) + dlat, min(lons) - dlon, max(lons) + dlon


def build_opentopo_url(demtype: str, south: float, north: float, west: float, east: float, api_key: str) -> str:
    params = {
        "demtype": demtype,
        "south": f"{south:.8f}",
        "north": f"{north:.8f}",
        "west": f"{west:.8f}",
        "east": f"{east:.8f}",
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    return f"{OPENTOPO_URL}?{urlencode(params)}"


def looks_like_geotiff(content: bytes, content_type: str) -> bool:
    if not content:
        return False
    tiff_magic = content.startswith(b"II*\x00") or content.startswith(b"MM\x00*")
    binary_type = any(token in content_type.lower() for token in ["tiff", "geotiff", "octet-stream", "application/x-tiff"])
    html_or_json = content.lstrip().startswith((b"<", b"{", b"["))
    return (tiff_magic or binary_type) and not html_or_json


def download_opentopo_dem(url: str, timeout=(10, 180)) -> bytes:
    if requests is None:
        raise RuntimeError("requests no está instalado.")
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        preview = re.sub(r"API_Key=[^&\s]+", "API_Key=****", r.text or "")[:500]
        raise RuntimeError(f"OpenTopography respondió HTTP {r.status_code}: {preview}")
    ctype = r.headers.get("Content-Type", "")
    if not looks_like_geotiff(r.content, ctype):
        preview = re.sub(r"API_Key=[^&\s]+", "API_Key=****", (r.text or ""))[:500]
        raise RuntimeError(f"La respuesta no parece GeoTIFF. Content-Type={ctype}. Respuesta: {preview}")
    return r.content


def read_dem_from_bytes(raw: bytes, name: str = "DEM descargado") -> Tuple[np.ndarray, Affine, CRS, Optional[float], Dict[str, Any]]:
    with MemoryFile(raw) as memfile:
        with memfile.open() as ds:
            arr = normalize_dem_array(ds.read(1), ds.nodata)
            meta = {
                "source_name": name,
                "width": ds.width,
                "height": ds.height,
                "crs": ds.crs.to_string() if ds.crs else None,
                "bounds": tuple(ds.bounds),
            }
            return arr, ds.transform, ds.crs, ds.nodata, meta

# -----------------------------------------------------------------------------
# Hidrología raster básica
# -----------------------------------------------------------------------------

def priority_flood_fill(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    nrows, ncols = dem.shape
    filled = np.array(dem, copy=True)
    visited = np.zeros(dem.shape, dtype=bool)
    heap: List[Tuple[float, int, int]] = []

    def push_if_valid(r: int, c: int):
        if 0 <= r < nrows and 0 <= c < ncols and valid[r, c] and not visited[r, c]:
            visited[r, c] = True
            heapq.heappush(heap, (float(filled[r, c]), r, c))

    for c in range(ncols):
        push_if_valid(0, c)
        push_if_valid(nrows - 1, c)
    for r in range(nrows):
        push_if_valid(r, 0)
        push_if_valid(r, ncols - 1)

    for r in range(1, nrows - 1):
        for c in range(1, ncols - 1):
            if not valid[r, c] or visited[r, c]:
                continue
            neigh_valid = [valid[r + dr, c + dc] for dr, dc in D8_OFFSETS]
            if not all(neigh_valid):
                push_if_valid(r, c)

    while heap:
        elev, r, c = heapq.heappop(heap)
        for dr, dc in D8_OFFSETS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < nrows and 0 <= cc < ncols):
                continue
            if not valid[rr, cc] or visited[rr, cc]:
                continue
            visited[rr, cc] = True
            if filled[rr, cc] < elev:
                filled[rr, cc] = elev
            heapq.heappush(heap, (float(filled[rr, cc]), rr, cc))

    filled[~valid] = np.nan
    return filled


def compute_d8_flow(filled_dem: np.ndarray, valid: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    nrows, ncols = filled_dem.shape
    flow_to = np.full((nrows, ncols), -1, dtype=np.int64)
    diag = math.hypot(res_x, res_y)
    distances = np.array([diag, res_y, diag, res_x, res_x, diag, res_y, diag], dtype="float64")

    for r in range(nrows):
        for c in range(ncols):
            if not valid[r, c]:
                continue
            z = filled_dem[r, c]
            best_idx = -1
            best_score = 0.0
            current_linear = r * ncols + c
            for k, (dr, dc) in enumerate(D8_OFFSETS):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < nrows and 0 <= cc < ncols) or not valid[rr, cc]:
                    continue
                dz = z - filled_dem[rr, cc]
                if dz > 0:
                    score = dz / distances[k]
                elif abs(dz) <= 1e-10 and (rr * ncols + cc) < current_linear:
                    score = 1e-12
                else:
                    score = -1.0
                if score > best_score:
                    best_score = score
                    best_idx = rr * ncols + cc
            flow_to[r, c] = best_idx
    return flow_to


def compute_accumulation(flow_to: np.ndarray, valid: np.ndarray) -> np.ndarray:
    nrows, ncols = flow_to.shape
    n = nrows * ncols
    flat_to = flow_to.ravel()
    valid_flat = valid.ravel()

    indeg = np.zeros(n, dtype=np.int32)
    srcs = np.where(valid_flat & (flat_to >= 0))[0]
    tgts = flat_to[srcs]
    np.add.at(indeg, tgts, 1)

    acc = np.zeros(n, dtype=np.float64)
    acc[valid_flat] = 1.0
    queue = list(np.where(valid_flat & (indeg == 0))[0])
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        j = flat_to[i]
        if j >= 0:
            acc[j] += acc[i]
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(int(j))
    return acc.reshape((nrows, ncols))


def snap_outlet_to_accumulation(outlet_rc: Tuple[int, int], accumulation: np.ndarray, valid: np.ndarray, radius_cells: int) -> Tuple[int, int]:
    r, c = outlet_rc
    nrows, ncols = accumulation.shape
    r0, r1 = max(0, r - radius_cells), min(nrows, r + radius_cells + 1)
    c0, c1 = max(0, c - radius_cells), min(ncols, c + radius_cells + 1)
    win = accumulation[r0:r1, c0:c1]
    vwin = valid[r0:r1, c0:c1]
    if not np.any(vwin):
        raise ValueError("El punto de control cae fuera del DEM o sobre NoData.")
    masked = np.where(vwin, win, -np.inf)
    local = np.unravel_index(np.nanargmax(masked), masked.shape)
    return int(r0 + local[0]), int(c0 + local[1])


def upstream_mask(flow_to: np.ndarray, valid: np.ndarray, outlet_rc: Tuple[int, int]) -> np.ndarray:
    nrows, ncols = flow_to.shape
    n = nrows * ncols
    outlet_idx = outlet_rc[0] * ncols + outlet_rc[1]
    flat_to = flow_to.ravel()
    valid_idx = np.where(valid.ravel() & (flat_to >= 0))[0]
    targets = flat_to[valid_idx]

    order = np.argsort(targets)
    targets_sorted = targets[order]
    sources_sorted = valid_idx[order]

    mask_flat = np.zeros(n, dtype=bool)
    stack = [int(outlet_idx)]
    mask_flat[outlet_idx] = True

    while stack:
        cur = stack.pop()
        left = np.searchsorted(targets_sorted, cur, side="left")
        right = np.searchsorted(targets_sorted, cur, side="right")
        if right <= left:
            continue
        for src in sources_sorted[left:right]:
            if not mask_flat[src]:
                mask_flat[src] = True
                stack.append(int(src))
    return mask_flat.reshape((nrows, ncols))


def polygon_from_mask(mask: np.ndarray, transform: Affine):
    geoms = []
    mask_uint = mask.astype("uint8")
    for geom, value in shapes(mask_uint, mask=mask, transform=transform):
        if value == 1:
            geoms.append(shape(geom))
    if not geoms:
        raise ValueError("No se pudo construir el polígono de cuenca.")
    return unary_union(geoms)


def compute_stream_mask(accumulation: np.ndarray, basin_mask: np.ndarray, threshold_cells: int) -> np.ndarray:
    return basin_mask & (accumulation >= max(1, int(threshold_cells)))


def stream_lengths(stream_mask: np.ndarray, flow_to: np.ndarray, transform: Affine, res_x: float, res_y: float) -> Tuple[float, int]:
    nrows, ncols = stream_mask.shape
    total = 0.0
    count = 0
    for r, c in zip(*np.where(stream_mask)):
        j = flow_to[r, c]
        if j < 0:
            continue
        rr, cc = divmod(int(j), ncols)
        if not (0 <= rr < nrows and 0 <= cc < ncols) or not stream_mask[rr, cc]:
            continue
        dist = math.hypot(res_x, res_y) if (rr != r and cc != c) else (res_y if rr != r else res_x)
        total += dist
        count += 1
    return total, count


def trace_main_channel(stream_mask: np.ndarray, flow_to: np.ndarray, accumulation: np.ndarray, outlet_rc: Tuple[int, int], res_x: float, res_y: float) -> float:
    nrows, ncols = stream_mask.shape
    flat_to = flow_to.ravel()
    valid_sources = np.where(stream_mask.ravel() & (flat_to >= 0))[0]
    targets = flat_to[valid_sources]
    order = np.argsort(targets)
    targets_sorted = targets[order]
    sources_sorted = valid_sources[order]

    outlet_idx = outlet_rc[0] * ncols + outlet_rc[1]
    best_len = 0.0
    stack = [(int(outlet_idx), 0.0)]
    visited = set()
    while stack:
        cur, length = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        best_len = max(best_len, length)
        left = np.searchsorted(targets_sorted, cur, side="left")
        right = np.searchsorted(targets_sorted, cur, side="right")
        if right <= left:
            continue
        candidate_sources = list(sources_sorted[left:right])
        candidate_sources.sort(key=lambda idx: accumulation.ravel()[idx], reverse=True)
        for src in candidate_sources[:8]:
            r1, c1 = divmod(int(cur), ncols)
            r0, c0 = divmod(int(src), ncols)
            dist = math.hypot(res_x, res_y) if (r0 != r1 and c0 != c1) else (res_y if r0 != r1 else res_x)
            stack.append((int(src), length + dist))
    return best_len


def longest_straight_basin_length(mask: np.ndarray, outlet_rc: Tuple[int, int], res_x: float, res_y: float) -> float:
    rr, cc = np.where(mask)
    if rr.size == 0:
        return float("nan")
    orow, ocol = outlet_rc
    dx = (cc - ocol) * res_x
    dy = (rr - orow) * res_y
    return float(np.nanmax(np.hypot(dx, dy)))


def basin_slope_stats(dem: np.ndarray, mask: np.ndarray, res_x: float, res_y: float) -> Tuple[float, float]:
    z = np.where(mask, dem, np.nan)
    gy, gx = np.gradient(z, res_y, res_x)
    slope = np.sqrt(gx ** 2 + gy ** 2) * 100.0
    return float(np.nanmean(slope)), float(np.nanmax(slope))


def compute_metrics(
    name: str,
    dem: np.ndarray,
    basin_mask: np.ndarray,
    basin_geom,
    stream_mask: np.ndarray,
    flow_to: np.ndarray,
    accumulation: np.ndarray,
    outlet_rc: Tuple[int, int],
    snapped_rc: Tuple[int, int],
    transform: Affine,
    res_x: float,
    res_y: float,
    cell_area_m2: float,
) -> Dict[str, Any]:
    area_m2 = float(np.sum(basin_mask) * cell_area_m2)
    area_km2 = area_m2 / 1_000_000.0
    area_ha = area_m2 / 10_000.0
    perim_m = float(basin_geom.length)

    z = dem[basin_mask]
    z = z[np.isfinite(z)]
    zmin = float(np.nanmin(z)) if z.size else float("nan")
    zmax = float(np.nanmax(z)) if z.size else float("nan")
    zmean = float(np.nanmean(z)) if z.size else float("nan")
    relief = zmax - zmin if np.isfinite(zmax) and np.isfinite(zmin) else float("nan")

    avg_slope_pct, max_slope_pct = basin_slope_stats(dem, basin_mask, res_x, res_y)
    total_stream_m, stream_segments = stream_lengths(stream_mask, flow_to, transform, res_x, res_y)
    main_len_m = trace_main_channel(stream_mask, flow_to, accumulation, snapped_rc, res_x, res_y)
    if not np.isfinite(main_len_m) or main_len_m <= 0:
        main_len_m = longest_straight_basin_length(basin_mask, snapped_rc, res_x, res_y)

    dd = (total_stream_m / 1000.0) / area_km2 if area_km2 > 0 else float("nan")
    basin_len_m = longest_straight_basin_length(basin_mask, snapped_rc, res_x, res_y)
    kc = 0.282 * perim_m / math.sqrt(area_m2) if area_m2 > 0 else float("nan")
    rcirc = 4.0 * math.pi * area_m2 / (perim_m ** 2) if perim_m > 0 else float("nan")
    ff = area_m2 / (basin_len_m ** 2) if basin_len_m > 0 else float("nan")
    re = (2.0 * math.sqrt(area_m2 / math.pi)) / basin_len_m if basin_len_m > 0 else float("nan")
    relief_ratio = relief / basin_len_m if basin_len_m > 0 and np.isfinite(relief) else float("nan")

    if main_len_m > 0 and relief > 0:
        # Kirpich: L en m, S adimensional. Tc en minutos.
        S = relief / main_len_m
        tc_min = 0.01947 * (main_len_m ** 0.77) * (S ** -0.385)
        tc_hr = tc_min / 60.0
    else:
        tc_min = float("nan")
        tc_hr = float("nan")

    ox, oy = xy_from_rc(transform, outlet_rc[0], outlet_rc[1])
    sx, sy = xy_from_rc(transform, snapped_rc[0], snapped_rc[1])

    return {
        "Nombre": name,
        "Área cuenca (km²)": area_km2,
        "Área cuenca (ha)": area_ha,
        "Área cuenca (m²)": area_m2,
        "Perímetro (km)": perim_m / 1000.0,
        "Cota mínima DEM cuenca (m)": zmin,
        "Cota máxima DEM cuenca (m)": zmax,
        "Cota media DEM cuenca (m)": zmean,
        "Relieve Hmax-Hmin (m)": relief,
        "Pendiente media raster (%)": avg_slope_pct,
        "Pendiente máxima raster (%)": max_slope_pct,
        "Longitud cauce principal aprox. (km)": main_len_m / 1000.0,
        "Longitud red drenaje aprox. (km)": total_stream_m / 1000.0,
        "Densidad drenaje aprox. (km/km²)": dd,
        "Longitud geométrica cuenca Lb (km)": basin_len_m / 1000.0,
        "Índice compacidad Gravelius Kc": kc,
        "Relación circularidad Rc": rcirc,
        "Factor forma Ff": ff,
        "Relación elongación Re": re,
        "Relación de relieve H/Lb": relief_ratio,
        "Tc Kirpich referencial (min)": tc_min,
        "Tc Kirpich referencial (h)": tc_hr,
        "Celdas de cauce usadas": int(np.sum(stream_mask)),
        "Segmentos de cauce usados": int(stream_segments),
        "Fila outlet original": outlet_rc[0],
        "Columna outlet original": outlet_rc[1],
        "Fila outlet ajustado": snapped_rc[0],
        "Columna outlet ajustado": snapped_rc[1],
        "X outlet original DEM CRS": ox,
        "Y outlet original DEM CRS": oy,
        "X outlet ajustado DEM CRS": sx,
        "Y outlet ajustado DEM CRS": sy,
        "Acumulación outlet ajustado (celdas)": float(accumulation[snapped_rc]),
    }


def delineate_watershed_from_precomputed(
    name: str,
    dem: np.ndarray,
    filled_dem: np.ndarray,
    valid: np.ndarray,
    transform: Affine,
    crs: CRS,
    flow_to: np.ndarray,
    accumulation: np.ndarray,
    outlet_xy_dem_crs: Tuple[float, float],
    snap_radius_m: float,
    stream_threshold_cells: int,
) -> WatershedResult:
    res_x = abs(float(transform.a))
    res_y = abs(float(transform.e))
    cell_area_m2 = res_x * res_y
    outlet_rc = rc_from_xy(transform, outlet_xy_dem_crs[0], outlet_xy_dem_crs[1])
    if not (0 <= outlet_rc[0] < dem.shape[0] and 0 <= outlet_rc[1] < dem.shape[1]):
        raise ValueError(f"{name}: el punto de control cae fuera de la extensión del DEM.")
    if not valid[outlet_rc]:
        raise ValueError(f"{name}: el punto de control cae sobre una celda NoData del DEM.")

    radius_cells = max(0, int(round(snap_radius_m / max(res_x, res_y))))
    snapped_rc = snap_outlet_to_accumulation(outlet_rc, accumulation, valid, radius_cells)
    basin_mask = upstream_mask(flow_to, valid, snapped_rc)
    if np.sum(basin_mask) < 3:
        raise ValueError(f"{name}: la cuenca resultante tiene menos de 3 celdas. Revise punto, DEM o radio de ajuste.")

    stream_mask = compute_stream_mask(accumulation, basin_mask, stream_threshold_cells)
    basin_geom = polygon_from_mask(basin_mask, transform)
    metrics = compute_metrics(
        name=name,
        dem=dem,
        basin_mask=basin_mask,
        basin_geom=basin_geom,
        stream_mask=stream_mask,
        flow_to=flow_to,
        accumulation=accumulation,
        outlet_rc=outlet_rc,
        snapped_rc=snapped_rc,
        transform=transform,
        res_x=res_x,
        res_y=res_y,
        cell_area_m2=cell_area_m2,
    )
    return WatershedResult(
        name=name,
        dem=dem,
        filled_dem=filled_dem,
        valid=valid,
        transform=transform,
        crs=crs,
        cell_area_m2=cell_area_m2,
        res_x=res_x,
        res_y=res_y,
        outlet_rc=outlet_rc,
        snapped_rc=snapped_rc,
        flow_to=flow_to,
        accumulation=accumulation,
        basin_mask=basin_mask,
        stream_mask=stream_mask,
        basin_geom=basin_geom,
        metrics=metrics,
    )

# -----------------------------------------------------------------------------
# Geometría doble cuenca, caudales y exportaciones
# -----------------------------------------------------------------------------

def geom_to_lonlat(geom, src_crs: CRS):
    transformer = Transformer.from_crs(PyCRS.from_user_input(src_crs), PyCRS.from_epsg(4326), always_xy=True)
    return shp_transform(transformer.transform, geom)


def lonlat_to_geom(geom, dst_crs: CRS):
    transformer = Transformer.from_crs(PyCRS.from_epsg(4326), PyCRS.from_user_input(dst_crs), always_xy=True)
    return shp_transform(transformer.transform, geom)


def calculate_incremental_area(res_hidro: WatershedResult, res_desc: WatershedResult) -> Dict[str, Any]:
    geom_h = res_hidro.basin_geom
    geom_d = res_desc.basin_geom
    area_h = float(geom_h.area) / 1_000_000.0
    area_d = float(geom_d.area) / 1_000_000.0
    inter_geom = geom_d.difference(geom_h)
    inter_area_geom = float(inter_geom.area) / 1_000_000.0 if not inter_geom.is_empty else 0.0
    area_simple = area_d - area_h
    union_area = geom_d.union(geom_h).area
    intersection_area = geom_d.intersection(geom_h).area
    # En vez de contains estricto, calcular porcentaje de cuenca hidro contenida en descarga.
    pct_hidro_inside_desc = (intersection_area / geom_h.area * 100.0) if geom_h.area > 0 else float("nan")
    pct_overlap_desc = (intersection_area / geom_d.area * 100.0) if geom_d.area > 0 else float("nan")
    is_contained = pct_hidro_inside_desc >= 99.0
    warnings = []
    if area_simple < 0:
        warnings.append("El área de la cuenca de descarga es menor que la cuenca hidrológica. Revise el orden de los puntos de control.")
    if not is_contained:
        warnings.append("La cuenca hidrológica no queda completamente dentro de la cuenca de descarga. Puede existir inversión de puntos o snap incorrecto al cauce.")
    if abs(area_simple - inter_area_geom) > max(0.01, 0.05 * max(abs(area_simple), 1e-9)):
        warnings.append("La diferencia simple de áreas no coincide con la diferencia geométrica. Revise contención de cuencas.")
    return {
        "geom_incremental": inter_geom,
        "area_hidrologica_km2": area_h,
        "area_descarga_km2": area_d,
        "area_incremental_simple_km2": area_simple,
        "area_incremental_geometrica_km2": inter_area_geom,
        "pct_hidrologica_dentro_descarga": pct_hidro_inside_desc,
        "pct_descarga_superpuesta_hidrologica": pct_overlap_desc,
        "contained": is_contained,
        "warnings": warnings,
    }


def q_table(q_hidrologico: Optional[float], q_adicional_manual: Optional[float], modo: str, area_h: float, area_inc: float, km_aporte: Optional[float], descripcion: str) -> pd.DataFrame:
    rows = []
    if q_hidrologico is None or not np.isfinite(q_hidrologico):
        q_hidrologico = 0.0
    q_add = 0.0
    metodo = "Sin aporte adicional"
    if modo == "Manual":
        q_add = float(q_adicional_manual or 0.0)
        metodo = "Aporte adicional manual"
    elif modo == "Estimado por área incremental":
        if area_h > 0:
            q_add = float(q_hidrologico) * max(area_inc, 0.0) / area_h
        else:
            q_add = float("nan")
        metodo = "Q adicional = Q hidrológico × Área incremental / Área hidrológica"
    q_total = q_hidrologico + q_add if np.isfinite(q_add) else float("nan")
    rows.append({
        "Método": metodo,
        "Q hidrológico PC-HIDRO (m³/s)": q_hidrologico,
        "Q adicional (m³/s)": q_add,
        "Q total aguas abajo (m³/s)": q_total,
        "Km ingreso aporte": km_aporte if km_aporte is not None else "No indicado",
        "Descripción": descripcion,
    })
    return pd.DataFrame(rows)


def metrics_dataframe(res_h: WatershedResult, res_d: WatershedResult, inc: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for res in [res_h, res_d]:
        for k, v in res.metrics.items():
            rows.append({"Grupo": res.name, "Parámetro": k, "Valor": v})
    for k in [
        "area_hidrologica_km2",
        "area_descarga_km2",
        "area_incremental_simple_km2",
        "area_incremental_geometrica_km2",
        "pct_hidrologica_dentro_descarga",
        "pct_descarga_superpuesta_hidrologica",
        "contained",
    ]:
        rows.append({"Grupo": "Intercuenca", "Parámetro": k, "Valor": inc.get(k)})
    for i, w in enumerate(inc.get("warnings", []), start=1):
        rows.append({"Grupo": "Advertencias", "Parámetro": f"Advertencia {i}", "Valor": w})
    return pd.DataFrame(rows)


def geojson_bytes(features: List[Dict[str, Any]], crs: CRS) -> bytes:
    fc = {
        "type": "FeatureCollection",
        "name": "hidrosed_doble_cuenca_v1",
        "crs": {"type": "name", "properties": {"name": crs.to_string()}},
        "features": features,
    }
    return json.dumps(fc, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def make_feature(geom, props: Dict[str, Any]) -> Dict[str, Any]:
    clean_props = {}
    for k, v in props.items():
        if isinstance(v, (np.floating, float)):
            clean_props[k] = float(v) if np.isfinite(v) else None
        elif isinstance(v, (np.integer, int)):
            clean_props[k] = int(v)
        elif isinstance(v, (bool, str)) or v is None:
            clean_props[k] = v
        else:
            clean_props[k] = str(v)
    return {"type": "Feature", "properties": clean_props, "geometry": mapping(geom)}


def make_kml_package(
    res_h: WatershedResult,
    res_d: WatershedResult,
    inc: Dict[str, Any],
    pc_h: ControlPoint,
    pc_d: ControlPoint,
    contours_ll: Optional[List[Dict[str, Any]]] = None,
) -> bytes:
    geom_h_ll = geom_to_lonlat(res_h.basin_geom, res_h.crs)
    geom_d_ll = geom_to_lonlat(res_d.basin_geom, res_d.crs)
    geom_inc_ll = geom_to_lonlat(inc["geom_incremental"], res_d.crs) if inc["geom_incremental"] is not None else None

    styles = """
    <Style id="cuenca_hidro"><LineStyle><color>ff00ffff</color><width>3</width></LineStyle><PolyStyle><color>3300ffff</color></PolyStyle></Style>
    <Style id="cuenca_desc"><LineStyle><color>ffff0000</color><width>3</width></LineStyle><PolyStyle><color>220000ff</color></PolyStyle></Style>
    <Style id="intercuenca"><LineStyle><color>ff00aa00</color><width>3</width></LineStyle><PolyStyle><color>3300aa00</color></PolyStyle></Style>
    <Style id="pc_hidro"><IconStyle><color>ff00ffff</color><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href></Icon></IconStyle></Style>
    <Style id="pc_desc"><IconStyle><color>ffff0000</color><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/paddle/blu-circle.png</href></Icon></IconStyle></Style>
    <Style id="curva"><LineStyle><color>ff996633</color><width>1</width></LineStyle></Style>
    """
    placemarks = []
    placemarks.append(polygon_to_kml_placemarks(geom_d_ll, "Cuenca descarga / soporte", "cuenca_desc", "Cuenca topográfica amplia de soporte."))
    placemarks.append(polygon_to_kml_placemarks(geom_h_ll, "Cuenca hidrológica", "cuenca_hidro", "Subcuenca hidrológica de cálculo."))
    placemarks.append(polygon_to_kml_placemarks(geom_inc_ll, "Intercuenca incremental", "intercuenca", "Área = cuenca descarga - cuenca hidrológica."))
    if pc_h.lon is not None and pc_h.lat is not None:
        placemarks.append(point_to_kml(pc_h.lon, pc_h.lat, "PC-HIDRO", "pc_hidro", pc_h.source))
    if pc_d.lon is not None and pc_d.lat is not None:
        placemarks.append(point_to_kml(pc_d.lon, pc_d.lat, "PC-DESCARGA", "pc_desc", pc_d.source))
    if contours_ll:
        for obj in contours_ll:
            placemarks.append(line_to_kml(obj["line"], f"Curva {obj['elev']:.2f} m", "curva", "Curva generada desde DEM; uso cartográfico/de apoyo."))

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>HidroSed Doble Cuenca v1</name>
    {styles}
    {''.join(placemarks)}
  </Document>
</kml>"""
    return make_kmz_from_kml(kml.encode("utf-8"))


def make_excel_bytes(metrics_df: pd.DataFrame, q_df: pd.DataFrame, summary_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Resumen")
        metrics_df.to_excel(writer, index=False, sheet_name="Metricas")
        q_df.to_excel(writer, index=False, sheet_name="Caudales")
    return output.getvalue()


def make_result_zip(
    res_h: WatershedResult,
    res_d: WatershedResult,
    inc: Dict[str, Any],
    pc_h: ControlPoint,
    pc_d: ControlPoint,
    metrics_df: pd.DataFrame,
    q_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    contours_ll: Optional[List[Dict[str, Any]]],
    include_dem: bool,
) -> bytes:
    features = [
        make_feature(res_d.basin_geom, {"nombre": "Cuenca descarga / soporte", **res_d.metrics}),
        make_feature(res_h.basin_geom, {"nombre": "Cuenca hidrologica", **res_h.metrics}),
    ]
    if inc["geom_incremental"] is not None and not inc["geom_incremental"].is_empty:
        features.append(make_feature(inc["geom_incremental"], {"nombre": "Intercuenca incremental", **{k: v for k, v in inc.items() if k != "geom_incremental"}}))

    kmz = make_kml_package(res_h, res_d, inc, pc_h, pc_d, contours_ll)
    geojson = geojson_bytes(features, res_h.crs)
    metrics_csv = metrics_df.to_csv(index=False).encode("utf-8-sig")
    q_csv = q_df.to_csv(index=False).encode("utf-8-sig")
    excel = make_excel_bytes(metrics_df, q_df, summary_df)
    summary_json = json.dumps({
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "app": APP_TITLE,
        "crs": res_h.crs.to_string(),
        "intercuenca": {k: (float(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else v) for k, v in inc.items() if k != "geom_incremental"},
        "q": q_df.to_dict(orient="records"),
        "pc_hidro": pc_h.__dict__,
        "pc_descarga": pc_d.__dict__,
        "nota": "Curvas desde DEM global son de apoyo y no reemplazan topografía de terreno.",
    }, ensure_ascii=False, indent=2, default=str).encode("utf-8")

    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("01_doble_cuenca_intercuenca.kmz", kmz)
        zf.writestr("02_doble_cuenca_intercuenca.geojson", geojson)
        zf.writestr("03_metricas.csv", metrics_csv)
        zf.writestr("04_caudales.csv", q_csv)
        zf.writestr("05_resultados.xlsx", excel)
        zf.writestr("06_resumen_integracion_hidrosed.json", summary_json)
        if include_dem:
            zf.writestr("07_dem_procesado_utm.tif", write_dem_to_geotiff_bytes(res_h.dem, res_h.transform, res_h.crs))
    return bio.getvalue()

# -----------------------------------------------------------------------------
# Curvas de nivel desde DEM
# -----------------------------------------------------------------------------

def extract_contours_from_dem(
    dem: np.ndarray,
    transform: Affine,
    src_crs: CRS,
    clip_geom_src_crs,
    interval_m: float,
    max_lines: int,
    simplify_m: float,
) -> List[Dict[str, Any]]:
    if measure is None:
        raise RuntimeError("scikit-image no está instalado; no se pueden generar curvas de nivel.")
    if interval_m <= 0:
        raise ValueError("La equidistancia de curvas debe ser mayor que cero.")
    mask = np.ones(dem.shape, dtype=bool)
    if clip_geom_src_crs is not None and not clip_geom_src_crs.is_empty:
        # Rasterizar el polígono de recorte de forma simple usando rasterio.features.geometry_mask.
        from rasterio.features import geometry_mask
        mask = ~geometry_mask([mapping(clip_geom_src_crs)], transform=transform, invert=False, out_shape=dem.shape)
    arr = np.where(mask & np.isfinite(dem), dem, np.nan)
    zmin = float(np.nanmin(arr))
    zmax = float(np.nanmax(arr))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        return []
    start = math.floor(zmin / interval_m) * interval_m
    end = math.ceil(zmax / interval_m) * interval_m
    levels = np.arange(start, end + interval_m * 0.5, interval_m)
    results: List[Dict[str, Any]] = []
    transformer = Transformer.from_crs(PyCRS.from_user_input(src_crs), PyCRS.from_epsg(4326), always_xy=True)
    # Rellenar NaN para measure.find_contours; se evita extraer por borde de no-data.
    fill_value = zmin - 10 * interval_m
    arr2 = np.where(np.isfinite(arr), arr, fill_value)
    for elev in levels:
        if len(results) >= max_lines:
            break
        if elev <= zmin or elev >= zmax:
            continue
        contours = measure.find_contours(arr2, float(elev))
        for contour in contours:
            if len(results) >= max_lines:
                break
            if contour.shape[0] < 3:
                continue
            coords_src = []
            for row, col in contour:
                x, y = transform * (float(col), float(row))
                coords_src.append((x, y))
            line = LineString(coords_src)
            if line.length < max(1.0, simplify_m):
                continue
            if clip_geom_src_crs is not None and not clip_geom_src_crs.is_empty:
                inter = line.intersection(clip_geom_src_crs)
                if inter.is_empty:
                    continue
                geoms = []
                if isinstance(inter, LineString):
                    geoms = [inter]
                elif isinstance(inter, MultiLineString):
                    geoms = list(inter.geoms)
                else:
                    continue
            else:
                geoms = [line]
            for g in geoms:
                if len(results) >= max_lines:
                    break
                if simplify_m > 0:
                    g = g.simplify(simplify_m, preserve_topology=False)
                if g.is_empty or g.length <= 0 or len(g.coords) < 2:
                    continue
                g_ll = shp_transform(transformer.transform, g)
                results.append({"elev": float(elev), "line": g_ll, "length_m": float(g.length)})
    return results

# -----------------------------------------------------------------------------
# Visualización
# -----------------------------------------------------------------------------

def plot_double_result(res_h: WatershedResult, res_d: WatershedResult, inc: Dict[str, Any]):
    dem = res_d.dem
    transform = res_d.transform
    nrows, ncols = dem.shape
    extent = [transform.c, transform.c + transform.a * ncols, transform.f + transform.e * nrows, transform.f]

    fig, ax = plt.subplots(figsize=(11, 8))
    plot_dem = np.array(dem, copy=True)
    if np.all(~np.isfinite(plot_dem)):
        plot_dem = np.zeros_like(plot_dem)
    else:
        plot_dem[~np.isfinite(plot_dem)] = np.nanmedian(plot_dem)

    try:
        ls = LightSource(azdeg=315, altdeg=45)
        hill = ls.hillshade(plot_dem, vert_exag=1.3, dx=res_d.res_x, dy=res_d.res_y)
        ax.imshow(hill, extent=extent, origin="upper", alpha=0.65)
    except Exception:
        ax.imshow(plot_dem, extent=extent, origin="upper", alpha=0.65)

    # Máscaras suaves
    d_masked = np.ma.masked_where(~res_d.basin_mask, dem)
    h_masked = np.ma.masked_where(~res_h.basin_mask, dem)
    ax.imshow(d_masked, extent=extent, origin="upper", alpha=0.30)
    ax.imshow(h_masked, extent=extent, origin="upper", alpha=0.45)

    # Polígonos
    for geom, label, lw in [(res_d.basin_geom, "Cuenca descarga", 2.4), (res_h.basin_geom, "Cuenca hidrológica", 2.4)]:
        try:
            geoms = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
            first = True
            for g in geoms:
                x, y = g.exterior.xy
                ax.plot(x, y, linewidth=lw, label=label if first else None)
                first = False
        except Exception:
            pass
    if inc.get("geom_incremental") is not None and not inc["geom_incremental"].is_empty:
        try:
            geoms = [inc["geom_incremental"]] if isinstance(inc["geom_incremental"], Polygon) else list(inc["geom_incremental"].geoms)
            first = True
            for g in geoms:
                if hasattr(g, "exterior"):
                    x, y = g.exterior.xy
                    ax.plot(x, y, linewidth=1.6, linestyle="--", label="Intercuenca" if first else None)
                    first = False
        except Exception:
            pass

    # Drenaje referencial descarga
    rr, cc = np.where(res_d.stream_mask)
    xs, ys = [], []
    for r, c in zip(rr, cc):
        j = res_d.flow_to[r, c]
        if j < 0:
            continue
        r2, c2 = divmod(int(j), dem.shape[1])
        if not res_d.stream_mask[r2, c2]:
            continue
        x1, y1 = xy_from_rc(transform, int(r), int(c))
        x2, y2 = xy_from_rc(transform, int(r2), int(c2))
        xs.extend([x1, x2, np.nan])
        ys.extend([y1, y2, np.nan])
    if xs:
        ax.plot(xs, ys, linewidth=0.8, label="Drenaje referencial")

    for res, marker, label in [(res_h, "o", "PC-HIDRO ajustado"), (res_d, "s", "PC-DESCARGA ajustado")]:
        sx, sy = xy_from_rc(transform, res.snapped_rc[0], res.snapped_rc[1])
        ax.scatter([sx], [sy], s=60, marker=marker, label=label)

    ax.set_title("Doble cuenca: hidrológica, descarga e intercuenca")
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def format_metric_value(v):
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", ".")
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "No calculado"
        return f"{float(v):,.4f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return str(v)

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("🌊 HidroSed · Módulo Doble Cuenca v1")
st.caption("Cuenca hidrológica + cuenca de descarga + intercuenca incremental + caudal adicional opcional.")

with st.expander("Objetivo de esta primera versión", expanded=True):
    st.markdown(
        """
        Este módulo reemplaza provisionalmente la etapa de delimitación simple de HidroSed por una lógica de **dos puntos de control**:

        - **PC-HIDRO:** define la subcuenca usada para hidrología y caudal base.
        - **PC-DESCARGA:** define la cuenca topográfica de soporte y cierre del tramo hidráulico.
        - **Intercuenca:** se calcula como `cuenca descarga - cuenca hidrológica`.
        - **Caudal adicional:** puede no usarse, ingresarse manualmente o estimarse por proporción de área.

        En v1 las curvas de nivel se generan desde el DEM cargado/procesado. Para diseño hidráulico fino, se deberán agregar después KMZ topográfico, DXF o secciones Excel.
        """
    )

left, right = st.columns([0.95, 1.05])

with left:
    st.subheader("1. DEM común")
    dem_mode = st.radio(
        "Fuente DEM",
        ["Cargar DEM GeoTIFF", "Descargar DEM OpenTopography"],
        index=0,
        help="La delimitación de ambas cuencas debe usar un solo DEM común.",
    )

    dem_file = None
    api_key = ""
    demtype = DEFAULT_DEM
    bbox_buffer_km = 5.0
    max_ot_area = 2_500.0
    downloaded_raw = None

    if dem_mode == "Cargar DEM GeoTIFF":
        dem_file = st.file_uploader("DEM GeoTIFF común (.tif/.tiff)", type=["tif", "tiff"])
    else:
        st.info("Para descargar DEM se requieren puntos PC-HIDRO y PC-DESCARGA en KMZ/KML, porque el bbox se calcula desde ambos puntos.")
        api_key = st.text_input("API Key OpenTopography", type="password")
        demtype = st.selectbox("DEM global", ["COP30", "NASADEM", "SRTMGL1", "SRTMGL3", "COP90"], index=0)
        bbox_buffer_km = st.number_input("Buffer bbox alrededor de ambos puntos (km)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
        max_ot_area = st.number_input("Área máxima bbox permitida para descarga (km²)", min_value=10.0, max_value=25000.0, value=2500.0, step=100.0)

    st.subheader("2. Puntos de control")
    point_mode = st.radio("Modo de ingreso de puntos", ["KMZ/KML", "Manual UTM WGS84"], index=0)

    pc_h_file = pc_d_file = None
    manual = {}
    if point_mode == "KMZ/KML":
        pc_h_file = st.file_uploader("PC-HIDRO · punto de control hidrológico", type=["kmz", "kml"], key="pc_h")
        pc_d_file = st.file_uploader("PC-DESCARGA · punto descarga/cuenca soporte", type=["kmz", "kml"], key="pc_d")
        st.caption("El KML/KMZ normalmente almacena coordenadas lon/lat WGS84 aunque Google Earth lo muestre como UTM.")
    else:
        c1, c2 = st.columns(2)
        manual["h_x"] = c1.number_input("PC-HIDRO Este UTM", value=350000.0, step=100.0, format="%.3f")
        manual["h_y"] = c2.number_input("PC-HIDRO Norte UTM", value=6680000.0, step=100.0, format="%.3f")
        c3, c4 = st.columns(2)
        manual["d_x"] = c3.number_input("PC-DESCARGA Este UTM", value=355000.0, step=100.0, format="%.3f")
        manual["d_y"] = c4.number_input("PC-DESCARGA Norte UTM", value=6680000.0, step=100.0, format="%.3f")
        c5, c6 = st.columns(2)
        manual["zone"] = c5.number_input("Huso UTM", min_value=1, max_value=60, value=19, step=1)
        manual["hemi"] = c6.selectbox("Hemisferio", ["Sur", "Norte"], index=0)

    st.subheader("3. Parámetros D8")
    snap_radius_m = st.number_input("Radio de ajuste de cada punto al cauce acumulado (m)", min_value=0.0, value=300.0, step=50.0)
    stream_threshold_cells = st.number_input("Umbral de drenaje referencial (celdas)", min_value=1, value=500, step=50)
    max_cells = st.number_input("Límite de celdas DEM a procesar", min_value=10000, value=2_000_000, step=100000)

    st.subheader("4. Curvas de nivel opcionales")
    generate_contours = st.checkbox("Generar curvas de nivel dentro de cuenca descarga", value=False)
    contour_interval = st.number_input("Equidistancia curvas (m)", min_value=0.5, value=5.0, step=0.5, disabled=not generate_contours)
    contour_simplify = st.number_input("Simplificación curvas (m)", min_value=0.0, value=5.0, step=1.0, disabled=not generate_contours)
    max_contours = st.number_input("Máximo de segmentos de curvas", min_value=100, value=3000, step=100, disabled=not generate_contours)

with right:
    st.subheader("5. Caudal adicional opcional")
    q_hidrologico = st.number_input("Q hidrológico en PC-HIDRO (m³/s)", min_value=0.0, value=0.0, step=0.1, help="Puede dejarse en 0 si solo desea calcular áreas.")
    q_mode = st.radio("Tratamiento del aporte incremental", ["Sin aporte adicional", "Manual", "Estimado por área incremental"], index=0)
    q_manual = 0.0
    if q_mode == "Manual":
        q_manual = st.number_input("Q adicional manual (m³/s)", min_value=0.0, value=0.0, step=0.1)
    km_aporte = st.number_input("Km ingreso aporte al cauce, opcional", min_value=0.0, value=0.0, step=0.1)
    km_aporte_val = float(km_aporte) if km_aporte > 0 else None
    desc_aporte = st.text_input("Descripción aporte", value="Intercuenca o aporte lateral al cauce")

    st.subheader("6. Exportación")
    include_dem_in_zip = st.checkbox("Incluir DEM procesado UTM en ZIP", value=False)

    st.warning(
        "Si PC-HIDRO no queda dentro de PC-DESCARGA, el área incremental no debe usarse sin revisar el sentido del cauce, el snap o el DEM."
    )

run_btn = st.button("Ejecutar doble delimitación", type="primary", use_container_width=True)

if run_btn:
    try:
        with st.status("Preparando datos...", expanded=True) as status:
            # 1) Leer puntos previos en lon/lat si se requiere descarga DEM.
            pre_points_lonlat: List[Tuple[float, float]] = []
            if point_mode == "KMZ/KML":
                if pc_h_file is None or pc_d_file is None:
                    st.error("Debe cargar PC-HIDRO y PC-DESCARGA.")
                    st.stop()
                h_lon, h_lat, _ = parse_first_kml_point(pc_h_file)
                d_lon, d_lat, _ = parse_first_kml_point(pc_d_file)
                pre_points_lonlat = [(h_lon, h_lat), (d_lon, d_lat)]
            else:
                src_utm = utm_epsg(int(manual["zone"]), "S" if manual["hemi"] == "Sur" else "N")
                to_ll = Transformer.from_crs(PyCRS.from_user_input(src_utm), PyCRS.from_epsg(4326), always_xy=True)
                h_lon, h_lat = to_ll.transform(float(manual["h_x"]), float(manual["h_y"]))
                d_lon, d_lat = to_ll.transform(float(manual["d_x"]), float(manual["d_y"]))
                pre_points_lonlat = [(h_lon, h_lat), (d_lon, d_lat)]

            # 2) Leer o descargar DEM.
            if dem_mode == "Cargar DEM GeoTIFF":
                if dem_file is None:
                    st.error("Debe cargar un DEM GeoTIFF.")
                    st.stop()
                status.write("Leyendo DEM cargado...")
                dem, transform, crs, nodata, dem_meta = read_dem_from_upload(dem_file)
            else:
                if not api_key.strip():
                    st.error("Debe ingresar API Key de OpenTopography.")
                    st.stop()
                south, north, west, east = bbox_from_points(pre_points_lonlat, float(bbox_buffer_km))
                area_bbox = bbox_area_km2(south, north, west, east)
                if area_bbox > float(max_ot_area):
                    st.error(f"El bbox calculado tiene {area_bbox:,.0f} km² y supera el límite configurado de {max_ot_area:,.0f} km².")
                    st.stop()
                url = build_opentopo_url(demtype, south, north, west, east, api_key.strip())
                status.write(f"Descargando DEM {demtype} desde OpenTopography. Área bbox aprox.: {area_bbox:,.1f} km²")
                raw = download_opentopo_dem(url)
                downloaded_raw = raw
                dem, transform, crs, nodata, dem_meta = read_dem_from_bytes(raw, f"OpenTopography {demtype}")

            if crs is None:
                raise ValueError("El DEM no tiene CRS definido.")
            status.write(f"DEM leído: {dem.shape[0]} filas x {dem.shape[1]} columnas; CRS: {crs.to_string()}.")

            # 3) Reproyección si corresponde.
            status.write("Verificando proyección en metros...")
            dem, transform, crs, reproj_meta = reproject_dem_to_projected(dem, transform, crs)
            if reproj_meta.get("reprojected"):
                status.write(f"DEM reproyectado automáticamente a {crs.to_string()}.")
            else:
                status.write("DEM ya estaba proyectado en metros.")

            if dem.size > int(max_cells):
                raise ValueError(
                    f"El DEM procesado tiene {dem.size:,} celdas; supera el límite configurado de {int(max_cells):,}. "
                    "Use un DEM más recortado, aumente el límite o descargue con menor área."
                )

            # 4) Crear puntos transformados al CRS del DEM.
            if point_mode == "KMZ/KML":
                h_lon, h_lat, _ = parse_first_kml_point(pc_h_file)
                d_lon, d_lat, _ = parse_first_kml_point(pc_d_file)
                h_x, h_y = transform_point_xy(h_lon, h_lat, "EPSG:4326", crs)
                d_x, d_y = transform_point_xy(d_lon, d_lat, "EPSG:4326", crs)
                pc_h = ControlPoint("PC-HIDRO", h_x, h_y, h_lon, h_lat, f"KMZ/KML lon={h_lon:.8f}, lat={h_lat:.8f}")
                pc_d = ControlPoint("PC-DESCARGA", d_x, d_y, d_lon, d_lat, f"KMZ/KML lon={d_lon:.8f}, lat={d_lat:.8f}")
            else:
                src_utm = utm_epsg(int(manual["zone"]), "S" if manual["hemi"] == "Sur" else "N")
                h_x, h_y = transform_point_xy(float(manual["h_x"]), float(manual["h_y"]), src_utm, crs)
                d_x, d_y = transform_point_xy(float(manual["d_x"]), float(manual["d_y"]), src_utm, crs)
                # lon/lat para KML
                to_ll = Transformer.from_crs(PyCRS.from_user_input(src_utm), PyCRS.from_epsg(4326), always_xy=True)
                h_lon, h_lat = to_ll.transform(float(manual["h_x"]), float(manual["h_y"]))
                d_lon, d_lat = to_ll.transform(float(manual["d_x"]), float(manual["d_y"]))
                pc_h = ControlPoint("PC-HIDRO", h_x, h_y, h_lon, h_lat, f"Manual {src_utm}: E={manual['h_x']:.3f}, N={manual['h_y']:.3f}")
                pc_d = ControlPoint("PC-DESCARGA", d_x, d_y, d_lon, d_lat, f"Manual {src_utm}: E={manual['d_x']:.3f}, N={manual['d_y']:.3f}")

            # 5) Hidrología precomputada una vez.
            res_x = abs(float(transform.a))
            res_y = abs(float(transform.e))
            valid = np.isfinite(dem)
            if np.sum(valid) < 10:
                raise ValueError("El DEM no contiene suficientes celdas válidas.")
            status.write("Rellenando depresiones con Priority-Flood...")
            filled_dem = priority_flood_fill(dem, valid)
            status.write("Calculando dirección de flujo D8...")
            flow_to = compute_d8_flow(filled_dem, valid, res_x, res_y)
            status.write("Calculando acumulación de flujo...")
            accumulation = compute_accumulation(flow_to, valid)

            # 6) Delimitar dos cuencas.
            status.write("Delimitando cuenca hidrológica PC-HIDRO...")
            res_h = delineate_watershed_from_precomputed(
                "Cuenca hidrológica PC-HIDRO",
                dem, filled_dem, valid, transform, crs, flow_to, accumulation,
                (pc_h.x_dem, pc_h.y_dem), float(snap_radius_m), int(stream_threshold_cells),
            )
            status.write("Delimitando cuenca descarga PC-DESCARGA...")
            res_d = delineate_watershed_from_precomputed(
                "Cuenca descarga PC-DESCARGA",
                dem, filled_dem, valid, transform, crs, flow_to, accumulation,
                (pc_d.x_dem, pc_d.y_dem), float(snap_radius_m), int(stream_threshold_cells),
            )

            # 7) Intercuenca y caudal.
            status.write("Calculando intercuenca incremental...")
            inc = calculate_incremental_area(res_h, res_d)
            q_df = q_table(
                q_hidrologico=float(q_hidrologico),
                q_adicional_manual=float(q_manual),
                modo=q_mode,
                area_h=inc["area_hidrologica_km2"],
                area_inc=inc["area_incremental_geometrica_km2"],
                km_aporte=km_aporte_val,
                descripcion=desc_aporte,
            )

            contours_ll = None
            if generate_contours:
                status.write("Generando curvas de nivel desde DEM dentro de cuenca descarga...")
                contours_ll = extract_contours_from_dem(
                    dem, transform, crs, res_d.basin_geom,
                    float(contour_interval), int(max_contours), float(contour_simplify),
                )
                status.write(f"Curvas generadas: {len(contours_ll)} segmentos.")

            metrics_df = metrics_dataframe(res_h, res_d, inc)
            summary_df = pd.DataFrame([
                {"Parámetro": "Área cuenca hidrológica PC-HIDRO (km²)", "Valor": inc["area_hidrologica_km2"]},
                {"Parámetro": "Área cuenca descarga PC-DESCARGA (km²)", "Valor": inc["area_descarga_km2"]},
                {"Parámetro": "Área incremental simple (km²)", "Valor": inc["area_incremental_simple_km2"]},
                {"Parámetro": "Área incremental geométrica (km²)", "Valor": inc["area_incremental_geometrica_km2"]},
                {"Parámetro": "% cuenca hidrológica dentro de descarga", "Valor": inc["pct_hidrologica_dentro_descarga"]},
                {"Parámetro": "Cuenca hidrológica contenida en descarga", "Valor": inc["contained"]},
                {"Parámetro": "Q hidrológico PC-HIDRO (m³/s)", "Valor": float(q_df.iloc[0]["Q hidrológico PC-HIDRO (m³/s)"])},
                {"Parámetro": "Q adicional (m³/s)", "Valor": float(q_df.iloc[0]["Q adicional (m³/s)"])},
                {"Parámetro": "Q total aguas abajo (m³/s)", "Valor": float(q_df.iloc[0]["Q total aguas abajo (m³/s)"])},
            ])
            zip_bytes = make_result_zip(res_h, res_d, inc, pc_h, pc_d, metrics_df, q_df, summary_df, contours_ll, bool(include_dem_in_zip))
            kmz_bytes = make_kml_package(res_h, res_d, inc, pc_h, pc_d, contours_ll)
            excel_bytes = make_excel_bytes(metrics_df, q_df, summary_df)
            geojson = geojson_bytes([
                make_feature(res_d.basin_geom, {"nombre": "Cuenca descarga", **res_d.metrics}),
                make_feature(res_h.basin_geom, {"nombre": "Cuenca hidrologica", **res_h.metrics}),
                make_feature(inc["geom_incremental"], {"nombre": "Intercuenca", **{k: v for k, v in inc.items() if k != "geom_incremental"}}) if inc["geom_incremental"] is not None and not inc["geom_incremental"].is_empty else None,
            ], crs)
            # Quitar None feature si intercuenca vacía
            features_clean = [f for f in [
                make_feature(res_d.basin_geom, {"nombre": "Cuenca descarga", **res_d.metrics}),
                make_feature(res_h.basin_geom, {"nombre": "Cuenca hidrologica", **res_h.metrics}),
                make_feature(inc["geom_incremental"], {"nombre": "Intercuenca", **{k: v for k, v in inc.items() if k != "geom_incremental"}}) if inc["geom_incremental"] is not None and not inc["geom_incremental"].is_empty else None,
            ] if f is not None]
            geojson = geojson_bytes(features_clean, crs)

            st.session_state["hidrosed_double_result"] = {
                "res_h": res_h,
                "res_d": res_d,
                "inc": inc,
                "pc_h": pc_h,
                "pc_d": pc_d,
                "metrics_df": metrics_df,
                "q_df": q_df,
                "summary_df": summary_df,
                "zip_bytes": zip_bytes,
                "kmz_bytes": kmz_bytes,
                "excel_bytes": excel_bytes,
                "geojson": geojson,
                "contours_count": len(contours_ll) if contours_ll else 0,
            }
            status.update(label="Proceso terminado", state="complete")

    except Exception as exc:
        st.exception(exc)
        st.stop()

# -----------------------------------------------------------------------------
# Resultados persistentes en session_state
# -----------------------------------------------------------------------------

if "hidrosed_double_result" in st.session_state:
    data = st.session_state["hidrosed_double_result"]
    res_h = data["res_h"]
    res_d = data["res_d"]
    inc = data["inc"]
    metrics_df = data["metrics_df"]
    q_df = data["q_df"]
    summary_df = data["summary_df"]

    st.divider()
    st.header("Resultados")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Área PC-HIDRO", f"{inc['area_hidrologica_km2']:.3f} km²")
    m2.metric("Área PC-DESCARGA", f"{inc['area_descarga_km2']:.3f} km²")
    m3.metric("Área incremental", f"{inc['area_incremental_geometrica_km2']:.3f} km²")
    m4.metric("Contención", f"{inc['pct_hidrologica_dentro_descarga']:.1f}%")

    if inc.get("warnings"):
        for w in inc["warnings"]:
            st.warning(w)
    else:
        st.success("Validación geométrica básica correcta: la cuenca hidrológica queda contenida en la cuenca de descarga.")

    st.subheader("Gráfico de control")
    st.pyplot(plot_double_result(res_h, res_d, inc))

    st.subheader("Resumen")
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Caudales")
    st.dataframe(q_df, use_container_width=True)

    with st.expander("Métricas completas", expanded=False):
        st.dataframe(metrics_df, use_container_width=True, height=520)

    st.subheader("Descargas")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button(
        "ZIP completo",
        data["zip_bytes"],
        file_name="hidrosed_modulo_doble_cuenca_v1_resultados.zip",
        mime="application/zip",
        use_container_width=True,
    )
    c2.download_button(
        "KMZ doble cuenca",
        data["kmz_bytes"],
        file_name="hidrosed_doble_cuenca_intercuenca.kmz",
        mime="application/vnd.google-earth.kmz",
        use_container_width=True,
    )
    c3.download_button(
        "Excel resultados",
        data["excel_bytes"],
        file_name="hidrosed_doble_cuenca_metricas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    c4.download_button(
        "GeoJSON",
        data["geojson"],
        file_name="hidrosed_doble_cuenca_intercuenca.geojson",
        mime="application/geo+json",
        use_container_width=True,
    )

st.divider()
st.markdown(
    """
    **Alcance v1:** delimitación doble con DEM común, intercuenca y caudal adicional opcional.  
    **Pendiente para v2:** eje hidráulico, riberas izquierda/derecha, secciones trapezoidales/rectangulares, carga de curvas KMZ de respaldo y carga de secciones Excel tipo HEC-RAS.
    """
)
