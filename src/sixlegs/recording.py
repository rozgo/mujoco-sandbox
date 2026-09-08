"""Render a saved dynamic trajectory as a multi-camera MP4."""
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sixlegs.scene import load_scene
from sixlegs.task import PHASES


def render_video(trajectory,output,speed=2.,fps=20):
    states=np.load(trajectory)
    if not len(states['time']):raise ValueError('Trajectory is empty')
    m,d=load_scene()
    option=mujoco.MjvOption();option.geomgroup[3]=0;option.sitegroup[:]=0
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    try:
        font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',22)
    except OSError:
        font=ImageFont.load_default(size=22)
    writer=imageio_ffmpeg.write_frames(str(output),(1280,1024),fps=fps,codec='libx264',quality=7,
                                       macro_block_size=16,output_params=['-movflags','+faststart'])
    writer.send(None)
    sample_times=np.arange(0,float(states['time'][-1]),speed/fps)
    camera_names=('head','left_wrist','right_wrist')
    with mujoco.Renderer(m,720,1280) as large, mujoco.Renderer(m,240,424) as small:
        try:
            for i,t in enumerate(sample_times):
                index=min(int(np.searchsorted(states['time'],t)),len(states['time'])-1)
                d.qpos[:]=states['qpos'][index];d.ctrl[:]=states['ctrl'][index]
                mujoco.mj_forward(m,d)
                large.update_scene(d,camera='third_person',scene_option=option)
                frame=Image.new('RGB',(1280,1024),'#101820')
                frame.paste(Image.fromarray(large.render()),(0,40))
                draw=ImageDraw.Draw(frame)
                phase=PHASES[int(states['phase'][index])].name
                draw.text((18,8),f'SIXLEGS  /  {t:05.1f}s  /  {phase}  /  {speed:g}x playback',font=font,fill='#e9f3f6')
                for j,camera in enumerate(camera_names):
                    small.update_scene(d,camera=camera,scene_option=option)
                    frame.paste(Image.fromarray(small.render()),(j*428,784))
                    draw.text((j*428+12,757),camera.upper().replace('_',' '),font=font,fill='#7bd5d5')
                writer.send(np.asarray(frame))
                if i%200==0:print(f'Rendering video: {i}/{len(sample_times)} frames',flush=True)
        finally:
            writer.close()
    print(f'Saved {output}',flush=True)
