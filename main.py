import numpy as np

bboxes = [[4,4,7,7],[3,3,8,8],[1,1,10,10],[2,2,11,12]]

def Remove_overlapping_boxes(bboxes):
    bboxes = np.array(bboxes)
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    picked_boxes = []
    order = np.argsort(np.ones(len(bboxes)))
    areas = (x2 - x1) * (y2 - y1)
    while order.size > 0:
        index = order[-1]
        picked_boxes.append(bboxes[index])

        # 获取当前置信度最大的候选框与其他任意候选框的相交面积
        x11 = np.maximum(x1[index], x1[order[:-1]])    # 计算重叠区域左侧x1值
        y11 = np.maximum(y1[index], y1[order[:-1]])    # 计算重叠区域左侧y1值
        x22 = np.minimum(x2[index], x2[order[:-1]])    # 计算重叠区域右侧x2值
        y22 = np.minimum(y2[index], y2[order[:-1]])    # 计算重叠区域右侧y2值

        # 计算当前矩形框与其余框的比值
        rate = areas[index] / areas[order[:-1]]               # 如果大于1，就表明比其余的框大，反之表明比其余的框小
        # 计算其余框于与当前框的比值
        rate1 = areas[order[:-1]] / areas[index]               # 这里是为了计算是否被某些框包含

        # 计算框与框之间相交的面积
        w = np.maximum(0.0, x22 - x11)
        h = np.maximum(0.0, y22 - y11)
        intersection = w * h

        # 利用相交的面积和两个框自身的面积计算框的交并比, 保留大于阈值的框
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        # rate==ratio表示包含关系，保留不为包含关系的框
        keep_boxes_indics = np.where(ratio != rate)
        keep_boxes_indics1 = np.where(ratio != rate1)                           # 保留不包含的框

        if keep_boxes_indics.__len__() < keep_boxes_indics1.__len__():
            order = order[keep_boxes_indics]
        else:
            order = order[keep_boxes_indics1]

    return_data = []
    for sig_picked_boxe in picked_boxes:
        x1 = sig_picked_boxe[0]
        y1 = sig_picked_boxe[1]
        x2 = sig_picked_boxe[2]
        y2 = sig_picked_boxe[3]
        return_data.append([x1,y1,x2,y2])
    return return_data
sig_bboxes = Remove_overlapping_boxes(bboxes)
print(sig_bboxes)