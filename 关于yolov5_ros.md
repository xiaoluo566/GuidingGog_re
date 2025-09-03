# 关于yolov5_ros

## 实现目的

实现了一个基于YOLOv5的ROS视觉检测系统，为导盲犬提供实时的环境感知能力

## ros包框架

📦  /
├── 📄 README.md                    # 英文说明文档
├── 📄 README_CN.md                 # 中文说明文档
└── 📁 yolov5_ros                  # 主要功能包目录
    ├── 📁 yolov5_ros              # 核心检测功能包
    │   ├── 📄 CMakeLists.txt       # ROS包构建配置
    │   ├── 📄 package.xml          # ROS包依赖声明
    │   ├── 📄 package.xml.bak      # 配置文件备份
    │   ├── 📁 launch              # ROS启动文件
    │   │   └── 📄 yolo_v5.launch   # 主启动配置
    │   ├── 📁 scripts             # Python执行脚本
    │   │   └── 📄 yolo_v5.py       # 核心检测脚本
    │   ├── 📁 weights             # 模型权重文件
    │   │   ├── 📄 download_weights.sh  # 权重下载脚本
    │   │   └── 📄 yolov5s.pt       # YOLOv5s预训练权重
    │   ├── 📁 media               # 媒体资源
    │   │   └── 📄 image.png        # 演示图片
    │   └── 📁 yolov5              # YOLOv5源码目录（仅列出主要文件）
    │       ├── 📄 detect.py        # 检测脚本
    │       ├── 📄 export.py        # 模型导出
    │       ├── 📄 requirements.txt # Python依赖
    │       ├── 📄 Dockerfile       # Docker配置
    │       ├── 📄 LICENSE          # 开源许可证
    │       └── 📁 utils           # 工具函数库
    └── 📁 yolov5_ros_msgs         # ROS消息定义包
        ├── 📄 CMakeLists.txt       # 消息包构建配置
        ├── 📄 package.xml          # 消息包依赖声明
        └── 📁 msg                 # 消息类型定义
            ├── 📄 BoundingBox.msg     # 单个检测框消息
            └── 📄 BoundingBoxes.msg # 多个检测框消息

## 代码解释

### yolov5_ros_msgs（ROS消息定义包）

是一个ROS消息定义包，专门用于定义和管理目标检测相关的自定义消息类型

其中msg文件夹下存放两个msg文件：BoundingBox.msg、BoundingBoxes.msg

#### BoundingBox.msg

描述目标检测算法检测到的**单个物体的边界框信息**

每检测到一个目标，就会生成一个 BoundingBox 消息，多个目标会组成 BoundingBoxes 消息进行发布

```
float64 probability   # 置信度，表示检测结果的可信程度（0~1之间的小数）
int64 xmin            # 边界框左上角的x坐标
int64 ymin            # 边界框左上角的y坐标
int64 xmax            # 边界框右下角的x坐标
int64 ymax            # 边界框右下角的y坐标
int16 num             # 目标类别编号
string Class          # 目标类别名称
```

#### BoundingBoxes.msg

描述一帧图像中**所有检测到的目标**的集合信息

```
Header header              # ROS标准消息头，包含时间戳、序列号、坐标系等元信息
Header image_header        # 原始图像的消息头，记录图像采集时的时间戳和坐标系
BoundingBox[] bounding_boxes  # 边界框数组，包含该帧图像中所有检测到的目标
```

### yolov5_ros（主要功能包）

#### scripts/yolo_v5.py

##### Yolo_Dect类

###### __init__

```py
def __init__(self):
```

先从ROS参数服务器获取配置

```python
yolov5_path = rospy.get_param('/yolov5_path', '')           # YOLOv5代码路径
weight_path = rospy.get_param('~weight_path', '')           # 权重文件路径
image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')  # 输入图像话题
pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')       # 输出话题
conf = rospy.get_param('~conf', '0.5')                     # 置信度阈值
```

加载本地yolov模型

```py
self.model = torch.hub.load(yolov5_path, 'custom',path=weight_path, source='local')
```

ROS通信设置

三种类型发布者

```py
# 订阅图像话题
self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,queue_size=1, buff_size=52428800)  # 大缓冲区防止图像丢失
# 参数 queue_size=1 为 消息队列=1
# 创建发布者
self.position_pub = rospy.Publisher('/yolov5/BoundingBoxes', BoundingBoxes, queue_size=1)
self.image_pub = rospy.Publisher('/yolov5/detection_image', Image, queue_size=1)
self.class_pub = rospy.Publisher('/yolov5/detected_classes', String, queue_size=1)
```

###### image_callback（图像回调处理函数）

```py
def image_callback(self, image):
```

```py
# 创建检测结果消息容器
self.boundingBoxes = BoundingBoxes() # 创建消息容器
self.boundingBoxes.header = image.header # 设置消息头
self.boundingBoxes.image_header = image.header # 保存图像头信息
# 图像数据转换
self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
# YOLOv5推理检测
results = self.model(self.color_image)
boxs = results.pandas().xyxy[0].values  # 获取检测框数据
# 处理检测结果（使用dectshow函数）
self.dectshow(self.color_image, boxs, image.height, image.width)
```

###### dectshow（检测结果处理函数）

```py
def dectshow(self, org_img, boxs, height, width):
```

**检测框处理循环**

```py
img = org_img.copy()
detected_classes = []  # 存储所有检测到的类别    
# 统计检测目标数量
count = len(boxs)  
for box in boxs:
    # 创建BoundingBox消息
    boundingBox = BoundingBox()
    boundingBox.probability = np.float64(box[4])  # 置信度
    boundingBox.xmin = np.int64(box[0])           # 左上角x
    boundingBox.ymin = np.int64(box[1])           # 左上角y
    boundingBox.xmax = np.int64(box[2])           # 右下角x
    boundingBox.ymax = np.int64(box[3])           # 右下角y
    boundingBox.num = np.int16(count)             # 总检测数
    boundingBox.Class = box[-1]                   # 类别名称   
    detected_classes.append(box[-1])              # 收集类别信息
```

**对检测结果进行绘制**

为每个类别分配颜色

```py
if box[-1] in self.classes_colors.keys():     # 检查类别是否已有颜色
    color = self.classes_colors[box[-1]]      # 已有则直接使用该颜色
else:
    color = np.random.randint(0, 183, 3)      # 随机生成颜色
    self.classes_colors[box[-1]] = color
```

绘制检测框

```py
cv2.rectangle(img, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])), (int(color[0]), int(color[1]), int(color[2])), 2)
```

智能标签位置计算

```py
if box[1] < 20:  # box[1]为左上角y
    text_pos_y = box[1] + 30  # 如果太靠近顶部，标签放在下方
else:
    text_pos_y = box[1] - 10  # 正常情况标签放在上方
```

绘制类别标签

```py
cv2.putText(img, box[-1],(int(box[0]), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
```

消息发布

```py
# 添加到检测结果集合
self.boundingBoxes.bounding_boxes.append(boundingBox)    
# 发布BoundingBoxes消息
self.position_pub.publish(self.boundingBoxes)    
# 发布检测类别列表（逗号分隔）
if detected_classes:
    class_msg = String()
    class_msg.data = ", ".join(detected_classes) 
    self.class_pub.publish(class_msg) 
# 发布可视化图像和显示
self.publish_image(img, height, width)
cv2.imshow('YOLOv5', img)
```

###### publish_image（图像发布函数）

```py
def publish_image(self, imgdata, height, width):
```

```py
image_temp = Image()  # 创建图像消息对象
header = Header(stamp=rospy.Time.now())  #创建消息头并设置时间戳（rospy.Time.now()可获取当前ros系统时间）
header.frame_id = self.camera_frame  # 设置坐标系ID
```

设置图像属性

```py
image_temp.height = height
image_temp.width = width
image_temp.encoding = 'bgr8'                    # 图像编码格式
image_temp.data = np.array(imgdata).tobytes()   # 图像数据转字节
image_temp.header = header
image_temp.step = width * 3                     # 每行字节数
```

##### main（主函数）

```py
def main():
    rospy.init_node('yolov5_ros', anonymous=True)  # 初始化ROS节点
    yolo_dect = Yolo_Dect()                        # 创建检测对象
    rospy.spin()                                   # 进入ROS循环
```

#### yolov5/detect.py

该脚本负责提供可靠的目标检测能力

用于对图像、视频、摄像头等多种输入源进行目标检测

##### run函数

```py
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
```

输入源判断

```py
source = str(source)  # 转为字符串
save_img = not nosave and not source.endswith('.txt')  # 是否保存推理结果的图像的条件
is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # Path(source).suffix[1:]为获取文件扩展名
is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)  # 判断 source 是否是一个实时流（如摄像头）或其他流媒体源（source为纯数字、为txt文件、是URL且不是文件）
```

创建保存目录

```py
save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment_path可以实现在目标路径已存在时自动递增目录名称
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
```

数据加载

```py
if webcam:  # 判断输入源是否为网络摄像头或流媒体
    view_img = check_imshow()  # 调用 check_imshow() 检查当前系统是否支持 cv2.imshow()（显示图像的功能）
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)  # 数据加载，并且参数调节大小
    bs = len(dataset)  # batch_size
else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
vid_path, vid_writer = [None] * bs, [None] * bs  # 初始化两个列表，用于保存视频路径和视频写入器
```

预热+数据预处理+推理

```py
model.warmup(imgsz=(1, 3, *imgsz), half=half)  # 预热
for path, im, im0s, vid_cap, s in dataset:
    t1 = time_sync()  # 记录当前时间
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1  # 数据预处理
    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    pred = model(im, augment=augment, visualize=visualize)  # 将预处理后的图像im输入模型，得到预测结果pred（是否用增强推理、是否保存可视化）
    t3 = time_sync()
    dt[1] += t3 - t2  # 计算推理时间
```

非极大值抑制

（对模型的原始预测结果进行后处理，移除重叠的边界框，保留置信度最高的框）

```py
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  
    dt[2] += time_sync() - t3  # 记录处理时间
```

结果处理

```py
# 遍历预测结果
for i, det in enumerate(pred):  # per image
    seen += 1
# 处理两种输入方式来选择路径（摄像头或者普通输入）
    if webcam:  # batch_size >= 1
        p, im0, frame = path[i], im0s[i].copy(), dataset.count
        s += f'{i}: '
    else:
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
# 设置保存路径
    p = Path(p)  # to Path
    save_path = str(save_dir / p.name)  # im.jpg
    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
# 记录图像尺寸信息
    s += '%gx%g ' % im.shape[2:]  # im.shape[2:]：获取图像的宽度和高度
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # gn为归一化因子，用于将检测框的坐标从模型输入尺寸映射回原始图像尺寸
    imc = im0.copy() if save_crop else im0  # for save_crop
# 初始化绘图工具
    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
# 检测结果处理
    if len(det):
         # Rescale boxes from img_size to im0 size
         det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
         # scale_coords：将检测框的坐标从模型输入尺寸映射回原始图像尺寸
# 打印检测结果
         for c in det[:, -1].unique():
             n = (det[:, -1] == c).sum()  # detections per class
             s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
         # Write results
# 保存检测结果
         for *xyxy, conf, cls in reversed(det):  # xyxy：检测框的坐标 [x1, y1, x2, y2] conf：置信度 cls：类别索引
             if save_txt:  # Write to file
                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                 line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                 with open(txt_path + '.txt', 'a') as f:
                     f.write(('%g ' * len(line)).rstrip() % line + '\n')
# 绘制检测框
             if save_img or save_crop or view_img:  # Add bbox to image
                 c = int(cls)  # integer class
                 label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                 annotator.box_label(xyxy, label, color=colors(c, True))  # box_label：在图像上绘制边框和标签
                 if save_crop:
                      save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
```

##### main函数

```py
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))  # 检查依赖项
    run(**vars(opt))  # 调用run函数进行推理
```
