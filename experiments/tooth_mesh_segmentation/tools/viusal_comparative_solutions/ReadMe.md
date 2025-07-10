```
可视化对比方案图
```
1. 观察单颌网格，保存相机姿态，保存为cam.json
python step1_take_pic_by_cam_parm.py xxx.ply
2. 根据 cam.json，截图
python step1_take_pic_by_cam_parm.py xxx.ply 2
3. 将多图拼接拼接为整图


meshes 筛选出的网格

render_mesh_[1-k]. k为最终图片的行数，即显示k个口扫网格 对网格复制n份（n个方案），并进行meshlab修改
render_pics_[1-k]. k为最终图片的行数，即显示k个口扫网格对网格复制n份（n个方案），并进行meshlab修改