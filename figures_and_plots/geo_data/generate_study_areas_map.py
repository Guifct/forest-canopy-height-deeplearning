import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection

# 1. Carregar contornos de Portugal
with open('portugal_osm.geojson', 'r', encoding='utf-8') as f:
    geo = json.load(f)

# Coordenadas ajustadas e retângulos proporcionais
study_areas = {
    'VPA': {
        'name': 'Vila Pouca de Aguiar (VPA)',
        'rect': (-7.82, 41.42, 0.26, 0.20),
        'text_pos': (-6.8, 41.52),
        'color': '#C0392B', # Vermelho
        'ha': 'left',
        'rad': 0.05
    },
    'SerraLousa': {
        'name': 'Serra da Lousã',
        'rect': (-8.32, 40.03, 0.30, 0.22),
        'text_pos': (-6.9, 40.14),
        'color': '#27AE60', # Verde
        'ha': 'left',
        'rad': 0.05
    },
    'Pombal': {
        'name': 'Pombal',
        'rect': (-8.75, 39.86, 0.25, 0.19),
        'text_pos': (-10.15, 40.05),
        'color': '#D35400', # Laranja
        'ha': 'right',
        'rad': -0.05
    },
    'Mafra': {
        'name': 'Mafra',
        'rect': (-9.40, 38.90, 0.18, 0.15),
        'text_pos': (-10.25, 39.22),
        'color': '#2980B9', # Azul
        'ha': 'right',
        'rad': -0.08
    },
    'SintraCascais': {
        'name': 'Sintra-Cascais',
        'rect': (-9.48, 38.68, 0.18, 0.15),
        'text_pos': (-10.25, 38.55),
        'color': '#8E44AD', # Roxo
        'ha': 'right',
        'rad': 0.06
    },
    'Monsanto': {
        'name': 'Monsanto (Lisboa)',
        'rect': (-9.22, 38.71, 0.15, 0.12),
        'text_pos': (-7.6, 38.77),
        'color': '#16A085', # Turquesa / Verde-azulado
        'ha': 'left',
        'rad': 0.05
    }
}

fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#F8FAFC')

# 2. Desenhar Portugal Continental
polygons = []
for poly in geo['coordinates']:
    ext_ring = poly[0]
    lons = [pt[0] for pt in ext_ring]
    lats = [pt[1] for pt in ext_ring]
    if min(lons) > -10.0 and max(lons) < -5.5 and min(lats) > 36.5 and max(lats) < 42.5:
        pts = [(pt[0], pt[1]) for pt in ext_ring]
        polygon = Polygon(pts, closed=True)
        polygons.append(polygon)

p_coll = PatchCollection(polygons, facecolor='#E2E8F0', edgecolor='#475569', linewidth=1.3, zorder=2)
ax.add_collection(p_coll)

# 3. Desenhar Retângulos e Callouts
for key, info in study_areas.items():
    rx, ry, rw, rh = info['rect']
    rect = Rectangle((rx, ry), rw, rh, linewidth=2.5, edgecolor=info['color'],
                     facecolor=info['color'], alpha=0.35, zorder=4)
    ax.add_patch(rect)
    
    rect_border = Rectangle((rx, ry), rw, rh, linewidth=2.5, edgecolor=info['color'],
                            facecolor='none', zorder=5)
    ax.add_patch(rect_border)

    cx = rx + rw / 2.0
    cy = ry + rh / 2.0
    tx, ty = info['text_pos']
    
    ax.annotate(
        info['name'],
        xy=(cx, cy),
        xytext=(tx, ty),
        fontsize=12,
        fontweight='bold',
        color='#0F172A',
        ha=info['ha'],
        va='center',
        zorder=6,
        bbox=dict(boxstyle='round,pad=0.45', facecolor='#FFFFFF', edgecolor=info['color'], linewidth=1.6, alpha=0.96),
        arrowprops=dict(
            arrowstyle='->,head_width=0.35,head_length=0.6',
            color=info['color'],
            lw=1.8,
            connectionstyle=f"arc3,rad={info.get('rad', 0.05)}"
        )
    )

# 4. Ajustar limites da vista
ax.set_xlim(-10.8, -5.7)
ax.set_ylim(36.7, 42.4)
ax.set_aspect(1.25)

# 5. Rosa dos Ventos reposicionada (canto superior esquerdo)
ax.annotate('N', xy=(-9.8, 41.8), xytext=(-9.8, 41.3),
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='#1E293B',
            arrowprops=dict(arrowstyle='-|>', facecolor='#1E293B', edgecolor='#1E293B', lw=2.2))

# 6. Barra de Escala (100 km)
scale_lon = -6.8
scale_lat = 37.1
km_len_deg = 0.90
ax.plot([scale_lon - km_len_deg, scale_lon], [scale_lat, scale_lat], color='#1E293B', lw=3.5, zorder=5)
ax.text(scale_lon - km_len_deg / 2, scale_lat + 0.07, '100 km', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1E293B')

# Grelha e formato de eixos
ax.grid(True, linestyle='--', alpha=0.4, color='#94A3B8', zorder=1)
ax.set_xlabel('Longitude (°W)', fontsize=11, fontweight='bold', color='#475569')
ax.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold', color='#475569')

import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{abs(v):.1f}°W"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, p: f"{v:.1f}°N"))

plt.title('Zonas de Estudo do Trabalho', fontsize=18, fontweight='bold', pad=20, color='#0F172A')
plt.tight_layout()

# Exportar
output_png = 'mapa_zonas_estudo.png'
output_pdf = 'mapa_zonas_estudo.pdf'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_pdf, bbox_inches='tight')
print("Mapa atualizado com sucesso!")
