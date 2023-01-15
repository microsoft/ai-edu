
class Meeting():
    start: time         # 开始时间
    end: time           # 结束时间
    host: People        # 主持人
    topic: str          # 议题或内容
    owner: People       # 订阅者
    status: Normal      # 状态(正常、进行中、取消、已结束)

class MeetingRoom():
    location: Geometry  # 位置
    no_room: str        # 会议室编号
    no_seat: int        # 座位数
    has_projetor: bool  # 有无投影仪
    has_whiteboard:bool # 有无白板
    has_network: bool   # 有无网络会议设备
