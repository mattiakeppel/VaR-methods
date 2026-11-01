"""Colours, typography and layout helpers shared by all slides."""
from pathlib import Path
import numpy as np
from manim import *

ROOT=Path(__file__).resolve().parent
BG="#FFFFFF"
INK="#202020"
MUTED="#414141"
NAVY=INK
RULE="#C6C6C6"
RETURNS="#B8B8B8"
ACCENT_COOL="#46677A"
ACCENT_WARM="#98604A"
TEMPLATE=TexTemplate()
TEMPLATE.add_to_preamble(r"\usepackage{lmodern}")
Tex.set_default(tex_template=TEMPLATE)
MathTex.set_default(tex_template=TEMPLATE)
config.background_color=BG
config.frame_width=16
config.frame_height=9
config.max_files_cached=1000

def tex_text(text):
    tokens={"Φ⁻¹":r"$\Phi^{-1}$","Φ":r"$\Phi$","σ":r"$\sigma$","μ":r"$\mu$",
            "→":r"$\rightarrow$","·":r"$\cdot$","–":"--","—":"---","−":"-",
            "\n":r"\\","&":r"\&","%":r"\%","_":r"\_","#":r"\#",
            "<":r"$<$",">":r"$>$","⇒":r"$\Rightarrow$"}
    result=[]; i=0
    while i<len(text):
        key=next((k for k in tokens if text.startswith(k,i)),None)
        result.append(tokens[key] if key else text[i]); i+=len(key) if key else 1
    return "".join(result)

def label(text,size=25,color=INK,bold=False):
    if not text: return VGroup()
    content=tex_text(text)
    if bold:content=r"\textbf{"+content+"}"
    content=(r"\shortstack[c]{"+content+"}") if "\n" in text else (r"\mbox{"+content+"}")
    return Tex(content,font_size=size,color=color)

BULLET_RADIUS=.045
BULLET_TEXT_GAP=.17
BULLET_ITEM_GAP=.24

def bullet_list(items,width,size=25,bold_labels=False,gap=BULLET_ITEM_GAP):
    """Filled circles with a hanging indent and natural line wrapping."""
    rows=VGroup()
    indent=2*BULLET_RADIUS+BULLET_TEXT_GAP
    for item in items:
        prefix=None
        if isinstance(item,tuple):
            item,prefix=item
        if bold_labels and prefix is None and ':' in item:
            prefix=item.split(':',1)[0]+':'
        content=tex_text(item)
        if prefix:
            content=r'\textbf{'+tex_text(prefix)+'}'+tex_text(item[len(prefix):])
        # dvisvgm converts a TeX point to .001037775 * font_size scene units.
        points=(width-indent)/(.001037775*size)
        body=Tex(r'\parbox[t]{'+f'{points:.4f}'+r'pt}{\raggedright '+content+r'\par}',
                 font_size=size,color=INK)
        left(body,indent,-body.height/2)
        mark=Dot(radius=BULLET_RADIUS,color=INK).move_to([BULLET_RADIUS,-min(.13,body.height/2),0])
        rows.add(VGroup(mark,body))
    return rows.arrange(DOWN,buff=gap,aligned_edge=LEFT)

def bullet_block(title,items,width=14.4,height=2.4,size=25,bold_labels=False,gap=BULLET_ITEM_GAP):
    card=Rectangle(width=width,height=height,stroke_width=0,fill_opacity=0)
    heading=left(label(title,26,INK,True),-width/2+.24,height/2-.31)
    rule=Line([-width/2+.24,height/2-.64,0],[width/2-.24,height/2-.64,0],color=RULE,stroke_width=.8)
    body=bullet_list(items,width-.48,size,bold_labels,gap)
    left(body,-width/2+.24,height/2-.84-body.height/2)
    return VGroup(card,rule,heading,body)

def fit(mob,width):
    if mob.width>width: mob.scale_to_fit_width(width)
    return mob

def left(mob,x,y):
    return mob.move_to([x+mob.width/2,y,0])

def eq(text,size=32):
    return MathTex(text,font_size=size,color=INK)

def block(title,lines,width=6.9,height=2.4,size=25):
    background=Rectangle(width=width,height=height,stroke_width=0,fill_opacity=0)
    bar=Line([-width/2+.24,height/2-.64,0],[width/2-.24,height/2-.64,0],color=RULE,stroke_width=.8)
    heading=left(fit(label(title,26,NAVY,True),width-.5),-width/2+.24,height/2-.31)
    body=VGroup(*[fit(label(t,size),width-.5) for t in lines])
    body.arrange(DOWN,buff=.16,aligned_edge=LEFT)
    if body.height>height-.9: body.scale_to_fit_height(height-.9)
    left(body,-width/2+.24,height/2-.83-body.height/2)
    return VGroup(background,bar,heading,body)

def panel(title,objects,width=14.4,height=2.0):
    """An unboxed section with stacked math and text."""
    card=Rectangle(width=width,height=height,stroke_width=0,fill_opacity=0)
    heading=left(label(title,25,NAVY,True),-width/2+.26,height/2-.32)
    body=VGroup(*objects).arrange(DOWN,buff=.10,aligned_edge=LEFT)
    fit(body,width-.55)
    if body.height>height-.86:
        body.scale((height-.86)/body.height)
    left(body,-width/2+.27,height/2-.75-body.height/2)
    rule=Line([-width/2+.26,height/2-.61,0],[width/2-.26,height/2-.61,0],color=RULE,stroke_width=.8)
    return VGroup(card,rule,heading,body)

def axes(xrange,yrange,width,height,left_x,bottom):
    a=Axes(x_range=xrange,y_range=yrange,x_length=width,y_length=height,tips=False,
           axis_config={"color":MUTED,"stroke_width":.9,"include_ticks":False})
    a.shift(np.array([left_x,bottom,0])-a.c2p(xrange[0],yrange[0]))
    return a

def line_on(ax,x,y,color=NAVY,width=1.6):
    return VMobject(stroke_color=color,stroke_width=width).set_points_as_corners([ax.c2p(float(a),float(b)) for a,b in zip(x,y)])

def note(text,y=-3.68,size=18):
    return fit(label(text,size,MUTED),14.4).move_to([0,y,0])

def legend_item(text,kind="solid",color=INK,size=21,stroke_width=None):
    if kind=="dash":
        mark=DashedLine(LEFT*.34,RIGHT*.34,color=color,stroke_width=2.0,dash_length=.10)
    elif kind=="dot":
        mark=Dot(radius=.045,color=color)
    elif kind=="circle":
        mark=Circle(radius=.055,color=color,stroke_width=1.4)
    else:
        mark=Line(LEFT*.34,RIGHT*.34,color=color,stroke_width=1.6)
    if stroke_width is not None:
        mark.set_stroke(width=stroke_width)
    return VGroup(mark,label(text,size)).arrange(RIGHT,buff=.14)
