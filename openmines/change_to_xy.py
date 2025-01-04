import math

# 地球半径 (米)
EARTH_RADIUS = 6371000.0

# 将度转换为弧度
def deg2rad(deg):
    return deg * (math.pi / 180)

# 计算两点之间的大圆距离
def haversine_distance(lat1, lon1, lat2, lon2):
    d_lat = deg2rad(lat2 - lat1)
    d_lon = deg2rad(lon2 - lon1)

    lat1 = deg2rad(lat1)
    lat2 = deg2rad(lat2)

    a = math.sin(d_lat / 2)**2 + math.sin(d_lon / 2)**2 * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return EARTH_RADIUS * c  # 返回距离（单位：米）

# 计算两点之间的方位角
def bearing(lat1, lon1, lat2, lon2):
    lat1 = deg2rad(lat1)
    lat2 = deg2rad(lat2)
    lon1 = deg2rad(lon1)
    lon2 = deg2rad(lon2)

    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return math.atan2(y, x)

# 经纬度转笛卡尔坐标
def convert_to_cartesian(lat0, lon0, lat, lon):
    dist = haversine_distance(lat0, lon0, lat, lon)  # 计算距离
    brng = bearing(lat0, lon0, lat, lon)  # 计算方位角

    x = dist * math.sin(brng)  # x 坐标
    y = dist * math.cos(brng)  # y 坐标

    return x, y

# 数据列表（经纬度信息）
points = [
    ("卸载区1", 30.7478335, 117.3908053),
    ("卸载区2", 30.74761699999999, 117.3904708),
    ("卸载点3", 30.7478551, 117.397711),
    ("卸载点4", 30.748091, 117.3977997),
    ("卸载点5", 30.7484041, 117.3982515),
    ("矿区", 30.74900629999999, 117.3947436),
    ("shovel1", 30.74811520874482, 117.3968462623751),
    ("shovel2", 30.75004224636199, 117.3963912953332),
    ("shovel3", 30.75097494245756, 117.3929901458248),
    ("shovel6", 30.74726357998347, 117.3879082851672),
    ("shovel4", 30.75146058473371, 117.390335058426),
    ("shovel5", 30.74976246642737, 117.3881715345722),
    ("充电站", 30.74617989735894, 117.3968376782259)
]

# 找到最小经纬度作为新原点
min_lat = min(point[1] for point in points)
min_lon = min(point[2] for point in points)

# 转换所有点
cartesian_points = []
for name, lat, lon in points:
    x, y = convert_to_cartesian(min_lat, min_lon, lat, lon)
    cartesian_points.append((name, x, y))

# 打印结果
print(f"新原点: Latitude = {min_lat}, Longitude = {min_lon}")
for name, x, y in cartesian_points:
    print(f"Name: {name}, x: {x:.2f} m, y: {y:.2f} m")
