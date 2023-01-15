
class Colors():
    Red = 1
    Green = 3

class Status():
    On = 1
    Off = 0

class Light():
    color: Colors
    status: Status 
    # 动作
    def TurnOn()
    def TurnOff()

class LightGroup():
    # 属性
    light_Straight: Light   # 直行灯
    light_LeftTurn: Light   # 左转灯
    # 动作
    def SetStraight(On|Off) # 允许|禁止直行
    def SetLeftTurn(On|Off) # 允许|禁止左转

class ControlCenter():
    light_groups: LightGroup[4] # 路口的每个方向都有一个灯组
    scheduler: time    # 定时器，用于对LightGroup进行开关操作
