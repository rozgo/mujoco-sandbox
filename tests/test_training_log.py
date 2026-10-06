from sixlegs.neural_insertion.training_log import boxes, history, summary

BOX = """╭──╮
│ PufferLib 5.0      \U0001F421 GPU:  11%   VRAM:   2.3/23G    RAM:  1.0G     │
│ Steps             {steps}      Env        2s 224  98%    value        0.010    │
│ SPS               {sps}      Copy          3ms   0%    entropy     -0.807    │
│ Epoch               {epoch}    Train           6ms   0%    old_kl       0.071    │
│ Uptime    {uptime}      Model         6ms   0%    kl           0.087    │
│ perf                          0.000   score                         2.337    │
│ collision                     0.066   timeout                       0.934    │
│ vessel_steps                  0.256   final_lateral_um           2249.950    │
│ final_vertical_um           132.206   level                         0.485    │
╰──╯"""


def test_dashboard_boxes_parse_into_numbers_and_compute():
    text = "\r".join([BOX.format(steps="131.1K", sps="27.0K", epoch=1, uptime="4s 859ms"),
                      BOX.format(steps="1.3M", sps="58.1K", epoch=10, uptime="23s 100ms"),
                      BOX.format(steps="37.6M", sps="45.4K", epoch=287, uptime="11m 2s 500ms"),
                      BOX.format(steps="37.7M", sps="14.8M", epoch=288, uptime="1h 0m 0s 0ms")])
    rows = boxes(text)
    assert [r["Steps"] for r in rows] == [131.1e3, 1.3e6, 37.6e6, 37.7e6]
    assert rows[2]["uptime_s"] == 662.5 and rows[3]["uptime_s"] == 3600
    assert rows[0]["kl"] == 0.087 and rows[0]["old_kl"] == 0.071 and rows[0]["entropy"] == -0.807
    assert rows[0]["gpu_pct"] == 11 and rows[0]["vram_gb"] == 2.3 and rows[0]["ram_gb"] == 1.0
    assert rows[0]["final_lateral_um"] == 2249.95 and rows[0]["level"] == 0.485
    s = summary(rows)
    assert s["steps_per_s"] == 37.7e6/3600  # wall clock, not the closing box's rate
    assert history(rows, every=2)[-1]["Steps"] == 37.7e6
