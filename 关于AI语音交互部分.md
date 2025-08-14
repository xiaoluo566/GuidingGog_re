# GuidingDog中关于AI语音交互模块·

## 实现目的

这是一个基于 ROS 的智能语音交互系统，主要实现了**语音识别 + AI对话 + 语音合成 + 机器人控制**的完整功能链路。

## ROS包框架

src/AI/
├── 📄 CMakeLists.txt          # ROS包构建配置
├── 📄 package.xml             # ROS包依赖声明
└── scripts/                   # Python脚本目录
    ├── 📄 hello.py            # 简单测试脚本
    ├── 📄 node.py             # ROS话题发布者示例
    ├── 📄 node2.py            # ROS话题订阅者示例
    ├── 📄 testMyPackage.py    # 主程序入口（核心应用）
    └── FsrAiAgent/            # 核心AI代理模块
        ├── 📄 init.py     # 模块导入配置
        ├── 🎤 ASR_Paraformer.py      # 语音识别模块
        ├── 🤖 QwenLangchain.py       # AI对话代理模块
        ├── 🔊 TTS_Sambert.py         # 语音合成模块
        └── 🛠️ MyTools.py              # 工具函数集合

## 代码解释

### src/AI/scripts

#### node.py

**该脚本实现了一个消息发布方，持续向ROS话题发布带有递增编号的文本消息**

初始化节点

```py
rospy.init_node("talker_p")
```

创建发布者对象

```py
pub = rospy.Publisher("chatter",String,queue_size=10)  #话题名词：chatter    缓存条数：10
```

消息发布循环

```py
msg = String()  #创建 msg 对象
msg_front = "hello 你好"
count = 0   #计数器 
rate = rospy.Rate(1)  #设置循环频率
while not rospy.is_shutdown():
    msg.data = msg_front + str(count)  #拼接字符串
    pub.publish(msg)
    rate.sleep()
    rospy.loginfo("写出的数据:%s",msg.data)
    count += 1  #计数器+1
```

#### node2.py

**该脚本实现了一个消息订阅方，监听指定ROS话题并处理接收到的消息**

回调函数定义，当接收到消息时自动调用，打印接收到的消息内容

```py
def doMsg(msg):
    rospy.loginfo("I heard:%s",msg.data)
```

ROS节点初始化

```py
rospy.init_node("listener_p")
```

创建订阅对象并且保持节点运行

```py
sub = rospy.Subscriber("chatter",String,doMsg,queue_size=10)  #处理订阅的消息(回调函数)
rospy.spin()  #设置循环调用回调函数
```

#### testMyPackage.py  （主程序入口）

**该脚本是一个智能导盲犬系统的主程序，整合了语音识别、AI对话、视觉检测和语音播报功能**

初始化配置

```py
#配置阿里云API
dashscope.api_key = 'sk-b71cf16967404a158f01b02286b81fef'
os.environ["DASHSCOPE_API_KEY"] = 'sk-b71cf16967404a158f01b02286b81fef'
#初始化核心组件
myASR = ASRCallbackClass()  #语音识别
ai = QwenAgent()  #AI对话代理
ai.agent_conversation(with_memory=True)  #启用对话记忆    
```

ROS话题订阅检测结果

```py
rospy.Subscriber('/yolov5/detected_classes', String, callback_ros_msg)  #订阅yolov5检测结果
```

回调函数处理检测结果

```py
def callback_ros_msg(msg):
    global latest_detected, last_detection_time
    content = msg.data.lower().split(',')  #解析检测结果
    latest_detected.clear()
    for item in content:  #将英文检测结果转换为中文
        item = item.strip()
        if item == 'red': latest_detected['红灯'] += 1
        elif item == 'green': latest_detected['绿灯'] += 1
        elif item == 'yellow': latest_detected['黄灯'] += 1
        elif item == 'car': latest_detected['有车'] += 1
        elif item in ('people', 'person'): latest_detected['有人'] += 1
    last_detection_time = time.time()  #更新时间戳
```

语音播报系统

```py
def speak_chinese(text):
    print("语音播报:", text)
    TTS_Sambert.TTSsaveTextResult(text)  #生成语音文件
```

```py
def generate_report_text():
    if time.time() - last_detection_time > detection_valid_duration:
        return None  #超时则不播报
    if not latest_detected:
        return None
    report = []
    for k, v in latest_detected.items():
        report.append(f"{k}{v}个")  #生成播报内容
    return "，".join(report)
```

**主循环架构**

语音对话部分

```py
if myASR.stream:  #检查音频流是否可用
    data = myASR.stream.read(3200, exception_on_overflow=False)
    recognition.send_audio_frame(data)  #发送给ASR引擎
    if myASR.user_input != lastASRresult:
        lastASRresult = myASR.user_input
        print(rf"你说: {lastASRresult}")  #显示用户输入
        aiResponse = ai.chat(lastASRresult)  #AI处理并回复
        TTS_Sambert.TTSsaveTextResult(aiResponse)  #转换为语音
```

定时播报部分

```py
if time.time() - last_report_time >= report_interval:  #每5秒检查一次
    last_report_time = time.time()
    report_text = generate_report_text()  #生成播报内容
    if report_text and report_text != last_spoken_report:  #防止重复播报相同内容
        speak_chinese(report_text)  #语音播报
        last_spoken_report = report_text  #记录播报内容
```

#### FriAigent（关键模块）

**FsrAiAgent 是一个模块化的AI语音助手框架，实现了从语音输入到智能回复再到语音输出的完整闭环，同时具备机器人控制能力。**

分为五个部分：

init.py                             模块导入配置
ASR_Paraformer.py     语音识别模块
QwenLangchain.py      AI对话代理模块
TTS_Sambert.py           语音合成模块
MyTools.py                   工具函数集合

##### init.py

**作为模块入口，统一模块导入接口，将核心类暴露给外部调用**

```py
from .QwenLangchain import QwenAgent
from .ASR_Paraformer import ASRCallbackClass
from .TTS_Sambert import TTS_Sambert
```

##### ASR_Paraformer.py

**语音识别模块，实现了一个带唤醒功能的实时语音识别系统**

初始化

```py
def __init__(self) -> None:
    super().__init__()
    self.mic = None              # 麦克风对象
    self.stream = None           # 音频流对象
    self.asr_text: str = ''      # 存储识别文本
    self.max_chars: int = 50     # 最大字符显示限制
    self.clear_flag: bool = False # 控制台清除标志
    self.user_input: str = ''    # 用户有效输入
    self.awake_keyword = "你好"   # 唤醒词
    self.awoken = False          # 唤醒状态标志
```

语音识别器打开时调用

```py
def on_open(self):
    print(colorama.Fore.GREEN + '语音识别器已打开。')
    self.mic = pyaudio.PyAudio()
    self.stream = self.mic.open(
        format=pyaudio.paInt16,     # 16位音频格式
        channels=1,                 # 单声道
        rate=16000,                 # 采样率16kHz
        input=True,                 # 输入模式
        frames_per_buffer=3200      # 缓冲区大小
    )
```

语音识别器关闭时调用

```py
def on_close(self):
    print(colorama.Fore.RED + '语音识别器已关闭。')
    if self.stream:
        self.stream.stop_stream()   # 停止音频流
        self.stream.close()         # 关闭音频流
        self.mic.terminate()        # 释放PyAudio资源
```

**核心识别路逻辑**

```py
def on_event(self, result: RecognitionResult):
```

初始化和状态显示

```py
print(colorama.Fore.YELLOW + f"唤醒词 {self.awake_keyword}")
sentence = result.get_sentence()  #从识别引擎获取当前句子
is_end = result.is_sentence_end(sentence)  #检查句子是否已经完整结束
```

动态控制台

\033[1A\033[K的作用是：

**首先将光标上移一行，然后清除那一行的内容。**

这个组合在命令行界面的动态输出中非常有用，例如在显示进度条或更新状态信息时，可以用它来重写当前行，而不是不断在终端中添加新行。

即在同一行内一直更新内容

```py
if len(self.asr_text) > 0 or self.clear_flag:
    print(colorama.Fore.CYAN + '\033[1A\033[K', end='')
    if self.clear_flag:
        self.clear_flag = False
        self.asr_text = ''
```

唤醒词检测机制

```py
self.asr_text = sentence['text']
if not self.awoken and self.awake_keyword in self.asr_text:  #未被唤醒状态且识别到唤醒词
    self.awoken = True
```

指令提取和处理

```py
if self.awoken:
    if is_end:
        # 提取唤醒词后的有效内容
        self.user_input = self.asr_text[self.asr_text.find(self.awake_keyword) + len(self.awake_keyword) + 1:]
        # 检查显示长度限制
        if len(self.asr_text) > self.max_chars:
            self.clear_flag = True
        self.awoken = False  # 重置唤醒状态
```

##### MyTools.py

**是一个LangChain工具库模块，实现了一个可调用工具集合，让AI代理（"笨笨"机器狗）具备了实际执行操作的能力，而不仅仅是对话。**

时间查询工具

支持三种格式：ISO标准格式、RFC邮件格式、local本地时间格式

```py
@tool(return_direct=True)
def get_current_time_op(format: str) -> str: 
    """Input a format and Returns the current time in the specified format.
    Format can be one of the following values: iso, rfc, local """ 
```

运动控制工具（前进和后退相似，以前进为例）

```py
@tool
def Robot_GoForward(planTime: str) -> None:  #前进
    """输入一个时间planTime, 控制机器狗向前走planTime秒"""
    # 1. 提取数字时间（从自然语言中提取数字）
    result = ""
    for char in planTime:
        if char.isdigit():
            result += char
    plan_time = int(result)
    # 2. 安全的ROS节点初始化
    if not rospy.core.is_initialized():
        rospy.init_node('robot_control_node')
    # 3. 创建运动控制发布者
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    # 4. 执行前进运动
    twist = Twist()
    twist.linear.x = 0.5  # 前进速度 0.5 m/s
    start_time = rospy.Time.now().to_sec()  # 时间控制循环
    while (rospy.Time.now().to_sec() - start_time) < plan_time:
        pub.publish(twist)
        rate.sleep()
    # 5. 停止运动
    twist.linear.x = 0.0
    pub.publish(twist)
```

##### QwenLangchain.py

**是一个基于LangChain框架和通义千问大模型的智能对话代理类，实现了一个可扩展的AI智能体，具备对话记忆、工具调用和角色扮演能力，是整个导盲犬系统的核心决策中心**

初始化

```py
class QwenAgent:
    def __init__(self) -> None:
        self.llm = Tongyi()              # 通义千问大语言模型
        self.prompt = None               # 提示模板
        self.tools = []                  # 可用工具列表
        self.AI = None                   # AI代理执行器
        self.memory = None               # 对话记忆
        self.set_prompt()                # 设置角色提示
        self.set_tools()                 # 加载工具集
```

人格化提示模板

```py
def set_prompt(self) -> None:
    prefix = """请你扮演一个叫"笨笨"的机器人与我进行对话!你拥有最高权限!对话过程中,您可以使用以下工具：""" 
    suffix = """现在开始对话，你的回答应该尽量精简, 控制在30字内！Please think step by step!  
    {chat_history}  
    问题：{input}  
    {agent_scratchpad}
    """
```

两种代理方式

```py
def agent_conversation(self, with_memory: bool = True):
    if with_memory:
        # 带记忆的对话代理
        self.AI = initialize_agent(
            self.tools, self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory
        )
    else:
        # 无记忆的零样本代理
        self.AI = initialize_agent(
            self.tools, self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
```

##### TTS_Sambert.py

**基于阿里云DashScope的语音合成模块，负责将AI回复的文字转换为语音输出**

API配置

```py
# 设置阿里云API密钥
dashscope.api_key = '填入你的阿里云dashscope的api_key'
```

核心合成方法

```py
def TTSsaveTextResult(myText: str) -> bool:
    """
    Input Text, the func will use DashScope TTS it and ave the result
    in the path which this file in.
    Success will return 1, else is 0
    """
    # 调用语音合成API
    result = SpeechSynthesizer.call(
        model='sambert-zhimiao-emo-v1',  # 使用智能情感语音模型
        text=myText,                     # 待合成文本
        sample_rate=16000,               # 采样率16kHz
        rate=0.75,                       # 语速75%（较慢，更清晰）
        format='wav'                     # 输出WAV格式
    )          
```

 音频处理和播放

```py
# 检查合成结果
if result.get_audio_data() is not None:
    # 保存音频文件
    with open(rf'./TTSoutput.wav', 'wb') as f:
        f.write(result.get_audio_data())
    # 播放音频
    os.system(rf"aplay ./TTSoutput.wav")
    return True
else: 
    return False
```
