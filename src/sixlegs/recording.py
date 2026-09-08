"""Render saved physical trajectories as labeled multi-camera MP4s."""
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sixlegs.scene import load_scene
from sixlegs.task import PHASES


def font(size):
    try:
        return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',size)
    except OSError:
        return ImageFont.load_default(size=size)


def completion_text(trajectory, final_time):
    report_path=Path(trajectory).with_name('transfer_report.json')
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report.get('success') and abs(report.get('simulation_seconds',0)-final_time)<.1:
            return 'TRANSFER COMPLETE  /  MUG + BLOCK PLACED AND RELEASED'
    return 'END OF RECORDED RUN'


def render_video(trajectory,output,speed=2.,fps=20,layout='overview'):
    """All-view layout: scene, following detail, overhead, head and both wrists."""
    if speed<=0 or fps<=0:raise ValueError('Speed and frame rate must be positive')
    states=np.load(trajectory)
    if not len(states['time']):raise ValueError('Trajectory is empty')
    m,d=load_scene()
    option=mujoco.MjvOption();option.geomgroup[3]=0;option.sitegroup[:]=0
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    all_views=layout=='all'
    size=(1920,1080) if all_views else (1280,1024)
    heading_font,label_font=font(28 if all_views else 22),font(22)
    writer=imageio_ffmpeg.write_frames(str(output),size,fps=fps,codec='libx264',quality=7,
        macro_block_size=2,output_params=['-movflags','+faststart','-threads','4'])
    writer.send(None)
    final_time=float(states['time'][-1])
    sample_times=np.arange(0,final_time,speed/fps)
    # Hold the verified final state long enough to inspect release and table support.
    sample_times=np.r_[sample_times,np.repeat(final_time,4*fps)]
    finish=completion_text(trajectory,final_time)
    main_shape=(452,640) if all_views else (720,1280)
    small_shape=(452,640) if all_views else (240,424)
    follow=mujoco.MjvCamera()
    follow.distance=3.8;follow.azimuth=135;follow.elevation=-27
    panels=[('third_person','01  SCENE'),(follow,'02  FOLLOWING DETAIL'),('overhead','03  OVERHEAD'),
            ('head','04  HEAD'),('left_wrist','05  LEFT WRIST / BLOCK'),('right_wrist','06  RIGHT WRIST / MUG')]
    with mujoco.Renderer(m,*main_shape) as large, mujoco.Renderer(m,*small_shape) as small:
        try:
            for i,t in enumerate(sample_times):
                index=min(int(np.searchsorted(states['time'],t)),len(states['time'])-1)
                d.qpos[:]=states['qpos'][index];d.ctrl[:]=states['ctrl'][index]
                mujoco.mj_forward(m,d)
                phase=PHASES[int(states['phase'][index])].name
                final=t>=final_time
                frame=Image.new('RGB',size,'#101820')
                draw=ImageDraw.Draw(frame)
                if all_views:
                    follow.lookat[:]=d.qpos[:3]+[.2,0,.15]
                    title=finish if final else f'SIXLEGS  /  {phase.upper()}'
                    draw.text((20,10),title,font=heading_font,fill='#e9f3f6')
                    draw.text((20,43),f'{t:05.1f} / {final_time:.1f} simulated seconds  |  {speed:g}x playback  |  Physical simulation',font=label_font,fill='#9bc4cc')
                    for j,(camera,label) in enumerate(panels):
                        x,y=(j%3)*640,72+(j//3)*484
                        large.update_scene(d,camera=camera,scene_option=option)
                        frame.paste(Image.fromarray(large.render()),(x,y+32))
                        draw.text((x+12,y+3),label,font=label_font,fill='#7bd5d5')
                    draw.rectangle((0,1040,1919,1047),fill='#263841')
                    draw.rectangle((0,1040,int(1919*t/final_time),1047),fill='#36c9ba')
                    draw.text((20,1053),'Six walking legs  /  Two Kinova Gen3 arms  /  Two Robotiq 2F-85 grippers',font=label_font,fill='#bdcdd4')
                else:
                    large.update_scene(d,camera='third_person',scene_option=option)
                    frame.paste(Image.fromarray(large.render()),(0,40))
                    label=finish if final else f'SIXLEGS  /  {t:05.1f}s  /  {phase}  /  {speed:g}x playback'
                    draw.text((18,8),label,font=heading_font,fill='#e9f3f6')
                    for j,camera in enumerate(('head','left_wrist','right_wrist')):
                        small.update_scene(d,camera=camera,scene_option=option)
                        frame.paste(Image.fromarray(small.render()),(j*428,784))
                        draw.text((j*428+12,757),camera.upper().replace('_',' '),font=label_font,fill='#7bd5d5')
                writer.send(np.asarray(frame))
                if i%100==0:print(f'Rendering video: {i}/{len(sample_times)} frames',flush=True)
        finally:
            writer.close()
    print(f'Saved {output}',flush=True)
