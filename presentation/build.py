"""Render the deck, then export final_presentation.html and final_presentation.pdf."""
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
import av
from PIL import Image

ROOT = Path(__file__).resolve().parent
PDF_SKIP = {"blank", "correct_pit_setup", "empirical_pit_setup",  # holding page and static PIT intros
            "tail_qq", "empirical_qq"}                              # covered by the combined / comparison Q-Q pages
COMBINED_QQ = ROOT/"media/images/presentation/combined_qq.png"     # replaces the correct_qq page in the PDF


def run(*args):
    subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True)


def last_frame(video_file):
    with av.open(str(video_file)) as video:
        stream = video.streams.video[0]
        video.seek(max(0, stream.duration - int(1.5 / float(stream.time_base))), stream=stream, backward=True)
        for frame in video.decode(video=0):
            image = frame.to_image()
    return image


def polish_html(path, links):
    """White player, no controls on the holding page, video preloading, R to replay,
    and invisible clickable areas over the credits on the closing slide."""
    anchors = "".join(
        f'<a href="{escape(link["url"], quote=True)}" aria-label="{escape(link["label"], quote=True)}" '
        f'title="{escape(link["label"], quote=True)}" target="_blank" rel="noopener noreferrer" '
        f'style="{";".join(f"{k}:{v:.5f}%" for k, v in link["bounds"].items())}"></a>' for link in links)
    extra = '''<style id="presentation-polish">
html,body,.reveal,.reveal .slide-background {background:white}
.reveal .controls{color:#202020}.reveal .progress{color:#202020}.reveal .slide-number{display:none}
body.holding .reveal .controls{display:none}
#closing-links{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);width:min(100vw,177.77778vh);height:min(100vh,56.25vw);z-index:10;pointer-events:none}
#closing-links[hidden]{display:none}
#closing-links a{position:absolute;display:block;pointer-events:auto;border-radius:2px}
#closing-links a:focus-visible{outline:1px solid #737373;outline-offset:5px}
</style>
<div id="closing-links" hidden aria-label="Presentation credits">''' + anchors + '''</div>
<script>
Reveal.on('ready',()=>{
  const links=document.getElementById('closing-links');
  const update=()=>{
    const current=Reveal.getCurrentSlide();
    document.body.classList.toggle('holding',current===Reveal.getSlides()[0]);
    links.hidden=current!==Reveal.getSlides().at(-1);
    const active=document.querySelector('.slide-background.present video');
    document.querySelectorAll('.slide-background video').forEach(video=>{
      video.playsInline=true;
      if(video!==active)video.pause();
    });
    if(active)active.preload='auto';
    const next=Reveal.getSlideBackground(Reveal.getIndices().h+1)?.querySelector('video');
    if(next){next.preload='auto';if(next.readyState===0)next.load();}
  };
  Reveal.on('slidechanged',()=>requestAnimationFrame(update));update();
  document.addEventListener('keydown',event=>{
    if(event.key.toLowerCase()==='r'&&!event.ctrlKey&&!event.metaKey&&!event.altKey){
      const video=document.querySelector('.slide-background.present video');
      if(video){video.currentTime=0;video.play().catch(()=>{});}
    }
  });
});
</script>'''
    html = path.read_text(encoding="utf-8")
    before, closing, after = html.rpartition("</body>")
    assert closing, "HTML export is missing its closing body tag"
    path.write_text(before + extra + closing + after, encoding="utf-8")


def vector_closing(path, links):
    """Replace the last (raster) PDF page with the vector closing slide plus link annotations.
    Pillow writes an uncompressed PDF, so an incremental update leaves the other pages untouched."""
    data = path.read_bytes()
    assert data.startswith(b"%PDF-") and len(re.findall(rb"startxref\s+\d+", data)) == 1
    objects = [(int(m[1]), m[2]) for m in re.finditer(rb"(\d+)\s+0\s+obj\b(.*?)endobj", data, re.S)]
    number, body = [(n, b) for n, b in objects if re.search(rb"/Type\s*/Page\b", b)][-1]
    parent = re.search(rb"/Parent\s+(\d+)\s+0\s+R", body)[1].decode()
    media = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", body)
    width, height = float(media[1]), float(media[2])
    trailer = data[data.rfind(b"trailer"):]
    size = int(re.search(rb"/Size\s+(\d+)", trailer)[1])
    root = re.search(rb"/Root\s+(\d+\s+0\s+R)", trailer)[1].decode()
    info = re.search(rb"/Info\s+(\d+\s+0\s+R)", trailer)
    previous = int(re.search(rb"startxref\s+(\d+)", trailer)[1])

    commands = [f"q {width/16} 0 0 {height/9} {width/2} {height/2} cm"]
    for p in ET.parse(ROOT/"media/closing_slide.svg").getroot().iter("{http://www.w3.org/2000/svg}path"):
        fill = float(p.get("fill-opacity", "1")) > 0
        stroke = float(p.get("stroke-opacity", "1")) > 0 and float(p.get("stroke-width", "0")) > 0
        if not fill and not stroke:
            continue
        for attr, op in (("fill", "rg"), ("stroke", "RG")):
            color = p.get(attr, "#202020").lstrip("#")
            commands.append(" ".join(f"{int(color[i:i+2], 16)/255:.6f}" for i in (0, 2, 4)) + " " + op)
        commands.append(p.get("stroke-width", "0") + " w")
        for kind, operands in re.findall(r"([MC])([^MC]+)", p.get("d", "")):
            commands.append(operands.strip() + " " + kind.lower())
        commands.append("B" if fill and stroke else "f" if fill else "S")
    stream = "\n".join(commands + ["Q"]).encode("ascii")

    updates = {size: b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"}
    annotations = []
    for i, link in enumerate(links, size + 1):
        b = link["bounds"]
        x0, y1 = b["left"]/100*width, height - b["top"]/100*height
        x1, y0 = x0 + b["width"]/100*width, y1 - b["height"]/100*height
        url = "(" + link["url"].replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"
        updates[i] = (f"<< /Type /Annot /Subtype /Link /Rect [{x0:.4f} {y0:.4f} {x1:.4f} {y1:.4f}] "
                      f"/Border [0 0 0] /A << /S /URI /URI {url} >> >>").encode("ascii")
        annotations.append(f"{i} 0 R")
    updates[number] = (f"<< /Type /Page /Parent {parent} 0 R /MediaBox [0 0 {width} {height}] "
                       f"/Resources << >> /Contents {size} 0 R /Annots [{' '.join(annotations)}] >>").encode()
    output = bytearray(data + b"\n")
    offsets = {}
    for obj, value in sorted(updates.items()):
        offsets[obj] = len(output)
        output.extend(f"{obj} 0 obj\n".encode() + value + b"\nendobj\n")
    xref = len(output)
    output.extend(b"xref\n")
    for obj, offset in sorted(offsets.items()):
        output.extend(f"{obj} 1\n{offset:010d} 00000 n \n".encode())
    info_ref = " /Info " + info[1].decode() if info else ""
    output.extend((f"trailer\n<< /Size {max(updates)+1} /Root {root}{info_ref} /Prev {previous} >>\n"
                   f"startxref\n{xref}\n%%EOF\n").encode())
    path.write_bytes(output)


def main():
    run("manim", "render", "-qh", "--fps", "60", "presentation.py", "Presentation")
    run("manim", "render", "-s", "-qh", "-o", "combined_qq", "presentation.py", "CombinedQQ")
    run("manim_slides", "convert", "Presentation", "final_presentation.html", "--one-file", "--offline",
        "-ctitle=VaR Forecasting and Diagnostic Validation", "-cbackground_color=white",
        "-creveal_version=6.0.1", "-creveal_theme=white", "-ccontrols=true", "-ccontrols_tutorial=false",
        "-cslide_number=false", "-ctransition=none", "-cbackground_transition=none", "-ctransition_speed=fast")
    links = json.loads((ROOT/"media/closing_links.json").read_text(encoding="utf-8"))
    polish_html(ROOT/"final_presentation.html", links)

    slides = json.loads((ROOT/"slides/Presentation.json").read_text(encoding="utf-8"))["slides"]
    keys = list(json.loads((ROOT/"notes.json").read_text(encoding="utf-8")))
    assert len(slides) == len(keys), (len(slides), len(keys))
    pages = [Image.open(COMBINED_QQ).convert("RGB") if key == "correct_qq" else last_frame(ROOT/s["file"])
             for s, key in zip(slides, keys) if key not in PDF_SKIP]
    pdf = ROOT/"final_presentation.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=144)
    vector_closing(pdf, links)
    print(f"Done: {len(slides)} slides, {len(pages)} PDF pages.")


if __name__ == "__main__":
    main()
