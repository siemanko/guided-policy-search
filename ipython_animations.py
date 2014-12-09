from IPython.display import HTML
from matplotlib import animation
from tempfile import NamedTemporaryFile
import sys
if sys.version_info[0] == 3:
    import base64
    encode_video = lambda x: base64.b64encode(x).decode("utf-8") 
else:
    encode_video = lambda x: x.encode("base64")


VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_video'):
        if not hasattr(anim, '_save_path'):
            with NamedTemporaryFile(suffix='.mp4') as f:
                anim.save(f.name, fps=20)
                video = open(f.name, "rb").read()
                anim._encoded_video = encode_video(video)
        else:
            video = open(anim._save_path, "rb").read()
            anim._encoded_video = encode_video(video)
    return VIDEO_TAG.format(anim._encoded_video)

def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))

def enable_inline():
    animation.Animation._repr_html_ = anim_to_html

