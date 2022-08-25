xyxyAreaFill = [1115, 830, 1670, 1375]   #[x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
xywhAreaFill = xyxyAreaFill   #[x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
xywhAreaFill[0] = xyxyAreaFill[0]   #x top-left
xywhAreaFill[1] = xyxyAreaFill[1]   #y top-left
xywhAreaFill[2] = xyxyAreaFill[2]-xyxyAreaFill[0]   #width
xywhAreaFill[3] = xyxyAreaFill[3]-xyxyAreaFill[1]   #height
areaSize = [3,3]
pointLocation = []
for j in range(areaSize[1]+1):
    for i in range(areaSize[0]+1):
        x = xywhAreaFill[2]/areaSize[0]*i + xywhAreaFill[0]
        y = xywhAreaFill[3]/areaSize[1]*j + xywhAreaFill[1]
        pointLocation.append([x,y])
print (pointLocation[1][0])
print(len(pointLocation))