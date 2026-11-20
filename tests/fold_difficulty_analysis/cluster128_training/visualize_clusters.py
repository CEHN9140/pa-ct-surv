"""Cluster overlay and eight single-cluster WSI panels; labels are slide-local."""
from pathlib import Path
import json
import h5py
import numpy as np
from PIL import Image,ImageDraw,ImageFont

COLORS=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#17becf']


def render_clusters(patient_id,feature_path,coord_path,labels,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    marker=destination/'geometry.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['labels_sha256']!=__import__('hashlib').sha256(labels.tobytes()).hexdigest():raise ValueError('Visualization labels changed')
        if (destination/'overview.png').exists() and (destination/'clusters.png').exists():return
    with h5py.File(coord_path,'r') as f:
        xy=f['coords'][:];attrs=dict(f['coords'].attrs)
    with h5py.File(feature_path,'r') as f:
        if not np.array_equal(xy,f['coords'][:]):raise ValueError('Feature/visualization coordinates differ')
    if len(xy)!=len(labels):raise ValueError('Coordinate/cluster count differs')
    width,height=int(attrs['level0_width']),int(attrs['level0_height']);footprint=float(attrs['patch_size_level0'])
    thumb=Path(coord_path).parents[2]/'thumbnails'/f'{patient_id}.jpg'
    if not thumb.exists():raise FileNotFoundError(thumb)
    with Image.open(thumb) as original:image=original.convert('RGB').copy()
    # The official thumbnail and level0 canvas must have the same aspect ratio up to rounding.
    if abs(image.width/image.height-width/height)>.01:raise ValueError('Thumbnail aspect ratio differs from WSI')
    image.thumbnail((1400,1000))
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,18) if Path(fontpath).exists() else ImageFont.load_default()
    groups,counts=np.unique(labels,return_counts=True)
    def overlay(base,only=None):
        base=base.convert('RGBA');layer=Image.new('RGBA',base.size);draw=ImageDraw.Draw(layer)
        sx,sy=base.width/width,base.height/height
        for (x,y),label in zip(xy,labels):
            if only is not None and label!=only:continue
            color=COLORS[int(label)%len(COLORS)];rgb=tuple(bytes.fromhex(color[1:]))
            draw.rectangle((x*sx,y*sy,(x+footprint)*sx,(y+footprint)*sy),fill=(*rgb,75),outline=(*rgb,200),width=1)
        return Image.alpha_composite(base,layer).convert('RGB')
    # Reserve sufficient header space for the two legend lines.
    overview=Image.new('RGB',(image.width,image.height+90),'white');d=ImageDraw.Draw(overview)
    d.text((8,3),f'{patient_id} | local clusters; colours have no cross-slide semantics',fill='black',font=font)
    for n,(g,count) in enumerate(zip(groups,counts)):
        xx=8+(n%4)*(image.width//4);yy=28+(n//4)*23
        d.text((xx,yy),f'C{g}: {count} ({count/len(labels):.0%})',fill=COLORS[int(g)],font=font)
    overview.paste(overlay(image),(0,82));overview.save(destination/'overview.png')
    panel=image.copy();panel.thumbnail((600,350));pw,ph=panel.size
    grid=Image.new('RGB',(4*pw,2*(ph+32)),'white');d=ImageDraw.Draw(grid)
    for n,g in enumerate(groups):
        xx=(n%4)*pw;yy=(n//4)*(ph+32)
        d.text((xx+6,yy+4),f'Cluster {g} | n={int((labels==g).sum())}',fill=COLORS[int(g)],font=font)
        grid.paste(overlay(panel,g),(xx,yy+32))
    grid.save(destination/'clusters.png')
    marker.write_text(json.dumps(dict(patient_id=str(patient_id),level0_width=width,level0_height=height,
        footprint_level0=footprint,n_patches=len(labels),labels_sha256=__import__('hashlib').sha256(labels.tobytes()).hexdigest(),
        note='Per-WSI unsupervised cluster IDs; not histological labels; colours are not comparable across WSIs'),indent=2))
