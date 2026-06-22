import cv2
from ultralytics import YOLO
import argparse
import sys
import os

# 强制开启 JIT 编译并屏蔽架构不兼容错误（针对最新的 RTX 50 系显卡）
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

  
from collections import defaultdict
import collections
import math
import numpy as np
from datetime import datetime
import time
import queue
import threading

class VoiceAnnouncer:
    def __init__(self):
        self.q = queue.Queue(maxsize=3)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'zh' in voice.languages or 'Chinese' in voice.name or 'zh' in voice.id.lower():
                    engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            print(f"语音播报初始化警告: {e}")
            
        while True:
            text = self.q.get()
            if text is None:
                break
            if engine:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
            self.q.task_done()

    def announce(self, text):
        cn_text = text.split(" / ")[0]
        # Ignore English parts for voice
        with self.q.mutex:
            self.q.queue.clear()
        try:
            self.q.put(cn_text, block=False)
        except queue.Full:
            pass

class AlertSmoother:
    def __init__(self):
        self.history = collections.deque(maxlen=15)
        self.current_state = "前方安全 / Safe"
        self.current_color = (0, 255, 0)
        self.last_announce_time = time.time()
        self.announcer = VoiceAnnouncer()
        
    def get_priority(self, text):
        if "STOP" in text or "Approaching" in text:
            return 4
        if "WAIT" in text or "Detour" in text:
            return 3
        if "Obstacle" in text or "Move" in text:
            return 2
        if "Notice" in text:
            return 1
        return 0

    def update(self, raw_text, raw_color):
        self.history.append((raw_text, raw_color))
        counts = collections.Counter([rt for rt, rc in self.history])
        
        best_text = self.current_state
        best_prio = -1
        
        for text, count in counts.items():
            prio = self.get_priority(text)
            # 紧急情况快速响应 (>= 2 帧)
            if prio >= 3 and count >= 2:
                if prio > best_prio:
                    best_prio = prio
                    best_text = text
            # 普通情况缓慢降级以去噪 (>= 10 帧)
            elif prio < 3 and count >= 10:
                if prio > best_prio:
                    best_prio = prio
                    best_text = text
                    
        # 找到对应颜色
        best_color = raw_color
        for t, c in self.history:
            if t == best_text:
                best_color = c
                break

        current_time = time.time()
        if best_text != self.current_state:
            self.current_state = best_text
            self.current_color = best_color
            self.announcer.announce(best_text)
            self.last_announce_time = current_time
        else:
            # 危险状态每3秒重复播报一次
            prio = self.get_priority(self.current_state)
            if prio >= 3 and current_time - self.last_announce_time > 3.0:
                self.announcer.announce(self.current_state)
                self.last_announce_time = current_time

        return self.current_state, self.current_color

def main():
    # 动态生成带时间戳的默认输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"output_demo_{timestamp}.mp4"

    parser = argparse.ArgumentParser(description="基于YOLOv8的导盲避障演示系统")
    parser.add_argument("--video", type=str, default="", help="输入视频的路径，不填则默认使用摄像头")
    parser.add_argument("--camera", type=int, default=0, help="摄像头设备索引，默认0。使用Iriun等虚拟摄像头时可以尝试1或2等")
    parser.add_argument("--output", type=str, default=default_output, help="输出视频文件的路径")
    args = parser.parse_args()

    # 加载YOLOv8s模型 (兼顾准确率与帧率)
    print("正在加载模型 yolov8s.pt ...")
    try:
        model = YOLO("yolov8s.pt")
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

    # 打开视频源 (优先使用视频，否则使用指定的摄像头索引)
    if args.video:
        source = args.video
    else:
        source = args.camera
        
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("无法打开视频源，请检查路径或摄像头状态。")
        sys.exit(1)

    # 获取视频属性用于保存预测结果
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 25  # 如果无法获取fps，则默认25

    # 设置视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print("开始处理视频... 按 'q' 推出。")

    # 定义画面左、中、右区域的边界（用于判断障碍物相对位置）
    left_bound = width // 3
    right_bound = 2 * (width // 3)

    # 历史轨迹字典，用于记录物体的横向坐标来计算速度
    track_history = defaultdict(list)
    
    # 告警平滑与语音播报器
    alert_smoother = AlertSmoother()

    # 镜头累计仿射矩阵，用于将背景作为“绝对参考系”（应对平移、旋转、缩放）
    prev_gray = None
    accumulated_matrix = np.eye(3, dtype=np.float32)

    while cap.isOpened():
        loop_start = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            print("视频 / 摄像头读取完毕。")
            break

        # ====== 引入整个背景作为超级“参考物”，计算摄像头自身的位移和旋转（包含竖向切变） ======
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            # 找到前一帧的特征点
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=10)
            if prev_pts is not None:
                # 用光流法在当前帧寻找这些特征点的位置
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
                if curr_pts is not None:
                    # 选出状态良好的匹配点
                    good_prev = prev_pts[status == 1]
                    good_curr = curr_pts[status == 1]
                    if len(good_prev) >= 3:
                        # [重磅升级] 计算仿射变换矩阵（自动计算并抵消 X/Y平移、所有方向的旋转倾斜、以及部分缩放距离）
                        # 求解从 当前帧 到 上一帧 的相对变换（利用RANSAC算法自动忽略掉屏幕中活动的其它行人车辆，只算大地路面等静止背景）
                        M, inliers = cv2.estimateAffinePartial2D(good_curr, good_prev)
                        if M is not None:
                            # 转换为 3x3 齐次矩阵以便累积历史的所有旋转平移
                            M_3x3 = np.vstack([M, [0, 0, 1]])
                            accumulated_matrix = accumulated_matrix @ M_3x3
                        
        prev_gray = gray.copy()
        # ==============================================================

        # 运行目标检测与追踪 (引入追踪以计算速度)
        results = model.track(source=frame, persist=True, conf=0.5, verbose=False)
        
        guidance_text = "前方安全 / Safe"
        guidance_color = (0, 255, 0) # 绿色
        
        # 解析预测结果
        for result in results:
            boxes = result.boxes
            # 找到最近/最具威胁的障碍物 (简单以面积大小作为距离的替代评估)
            max_area = 0
            danger_box = None
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                
                # 若检测到的物体面积占据屏幕的比例超过阈值，视为可能的障碍物
                if area > max_area:
                    max_area = area
                    danger_box = box

            if danger_box is not None:
                # 分析最危险边界框的位置
                x1, y1, x2, y2 = danger_box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 简单计算占据画面的比例来估测距离(越近越危险)
                area_ratio = max_area / (width * height)
                
                if area_ratio > 0.15:  # 面积大于画面15%认为距用户较近，具有威胁
                    # 判断是否为快速移动的物体（通过连续追踪多帧的位置和大小变化）
                    is_fast_moving = False
                    is_approaching = False
                    if danger_box.id is not None:
                        # 记录此物体的中心 (X, Y) 坐标及面积历史轨迹
                        obj_id = int(danger_box.id[0].cpu().numpy())
                        
                        # [重大升级] 坐标剥离与映射！应用累积的仿射矩阵
                        # 不仅扣除简单的平移，还要扣除摄像头低头仰头、旋转倾角甚至部分拉近缩放带来的屏幕透视影响
                        local_pt = np.array([center_x, center_y, 1.0])
                        global_pt = accumulated_matrix @ local_pt
                        real_x, real_y = global_pt[0], global_pt[1]
                        
                        track_history[obj_id].append((real_x, real_y, area))
                        
                        # 仅保留最近的10帧坐标
                        if len(track_history[obj_id]) > 10:
                            track_history[obj_id].pop(0)
                        
                        # 如果追踪该物体的帧数大于等于5帧，则开始计算它的综合位移距离
                        if len(track_history[obj_id]) >= 5:
                            dx = track_history[obj_id][-1][0] - track_history[obj_id][0][0]
                            dy = track_history[obj_id][-1][1] - track_history[obj_id][0][1]
                            moved_distance = math.hypot(dx, dy)
                            
                            # 计算面积增长率，判断是否朝向受试者快速逼近
                            area_diff = track_history[obj_id][-1][2] - track_history[obj_id][0][2]
                            area_growth = area_diff / track_history[obj_id][0][2] if track_history[obj_id][0][2] > 0 else 0
                            
                            if area_growth > 0.2: # 面积在短时间内显著增大，说明快速逼近
                                is_approaching = True
                            
                            # 判断位移是否显著（综合欧氏距离超过画面宽度的 8%）
                            if moved_distance > width * 0.08:
                                is_fast_moving = True
                    
                    # 计算障碍物两侧的剩余通行空间
                    left_space = x1              # 障碍物左侧到画面左边缘的距离
                    right_space = width - x2     # 障碍物右侧到画面右边缘的距离

                    if is_approaching:
                        # 如果物体正在快速逼近，则需要紧急躲避而不是等待
                        if left_space > right_space:
                            guidance_text = "物体快速逼近, 紧急向左躲避! / Approaching, Dodge Left!"
                        else:
                            guidance_text = "物体快速逼近, 紧急向右躲避! / Approaching, Dodge Right!"
                        guidance_color = (0, 0, 255) # 红色
                    elif is_fast_moving:
                        # 如果是高速横穿的物体，指导原地等待
                        guidance_text = "前方有快速移动物体, 请原地等待! / Fast Object, WAIT!"
                        guidance_color = (0, 0, 255) # 红色
                    else:
                        if center_x < left_bound:
                            guidance_text = "左方障碍, 建议向右方微调避让 / Obstacle L, Move Right"
                            guidance_color = (0, 165, 255) # 橙色
                        elif center_x > right_bound:
                            guidance_text = "右方障碍, 建议向左方微调避让 / Obstacle R, Move Left"
                            guidance_color = (0, 165, 255) # 橙色
                        else:
                            # 障碍物在正前方，此时计算哪边空间更大，指导用户绕行
                            if left_space > right_space and left_space > width * 0.25:
                                guidance_text = "正前方障碍, 左侧空间大, 请向左绕行 / Ahead, Detour Left"
                                guidance_color = (0, 0, 255) # 红色
                            elif right_space > left_space and right_space > width * 0.25:
                                guidance_text = "正前方障碍, 右侧空间大, 请向右绕行 / Ahead, Detour Right"
                                guidance_color = (0, 0, 255) # 红色
                            else:
                                guidance_text = "前方完全受阻, 无安全通道, 请停止! / Path Blocked, STOP!"
                                guidance_color = (0, 0, 255) # 红色
                elif area_ratio > 0.05:
                     guidance_text = "注意前方有物体 / Notice: Object Ahead"
                     guidance_color = (255, 255, 0) # 青色

        # === 运用平滑与播报模块，去除闪烁噪声 ===
        stable_text, stable_color = alert_smoother.update(guidance_text, guidance_color)

        # 将检测框画在原图上
        annotated_frame = results[0].plot()

        # 在画面顶部显示指导策略文字
        cv2.rectangle(annotated_frame, (0, 0), (width, 60), (0,0,0), -1)
        # OpenCVputText对于中文支持不好，此处使用英文提示或简单的拼音拼写，
        # 如果需要显示完美中文，可以借用PIL库(由于是Demo简化处理，此处展示英文或拼音)
        cv2.putText(annotated_frame, stable_text.split(" / ")[-1], (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, stable_color, 2, cv2.LINE_AA)

        # 计算并显示处理帧率(FPS)
        loop_end = cv2.getTickCount()
        fps_real = cv2.getTickFrequency() / (loop_end - loop_start)
        cv2.putText(annotated_frame, f"FPS: {int(fps_real)}", (width - 150, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 写入视频
        out.write(annotated_frame)

        # 实时显示（可选）
        cv2.imshow("Guide Demo", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成，输出视频已保存到: {args.output}")

if __name__ == "__main__":
    main()
