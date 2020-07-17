
def mixup():
    a = 0
    for imgname in img_names:
        a = a + 1
        imgpath = img_path + imgname
        img = cv2.imread(imgpath)
        img_h, img_w = img.shape[0], img.shape[1]
        i = random.randint(0, img_num - 1)
        print('i:', i)
        add_path = img_path + img_names[i]
        addimg = cv2.imread(add_path)
        add_h, add_w = addimg.shape[0], addimg.shape[1]
        if add_h != img_h or add_w != img_w:
            print("resize!")
            addimg = cv2.resize(addimg, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        scale_h, scale_w = img_h / add_h, img_w / add_w
        lam = np.random.beta(20, 20)
        mixed_img = lam * img + (1 - lam) * addimg
        save_img = save_path + imgname[:-8] + '_' + ('%d' % a) + '.jpg'
        cv2.imwrite(save_img, mixed_img)
        print("save_img:", save_img)

        if imgname != img_names[i]:  # 待增强的和加上去的那张图片不是同一张图片
            # 接下来要对标签进行读取与处理
            xmlfile1 = xml_path + imgname[:-4] + '.xml'
            xmlfile2 = xml_path + img_names[i][:-4] + '.xml'
            print("xml_1:", xmlfile1)
            print("xml_2:", xmlfile2)
            # 读取两张图片的xml
            tree1 = ET.parse(xmlfile1)
            tree2 = ET.parse(xmlfile2)
            doc = xml.dom.minidom.Document()
            root = doc.createElement("annotation")
            doc.appendChild(root)

            nodeframe = doc.createElement("frame")
            nodeframe.appendChild(doc.createTextNode(imgname[:-8] + ('%d' % a)))

            objects = []
            for obj in tree1.findall("object"):
                obj_struct = {}
                obj_struct["name"] = obj.find("name").text
                bbox = obj.find("bndbox")
                obj_struct["bbox"] = [int(bbox.find("xmin").text),
                                      int(bbox.find("ymin").text),
                                      int(bbox.find("xmax").text),
                                      int(bbox.find("ymax").text)]
                objects.append(obj_struct)

            for obj in tree2.findall("object"):
                obj_struct = {}
                obj_struct["name"] = obj.find("name").text
                bbox = obj.find("bndbox")
                obj_struct["bbox"] = [int(int(bbox.find("xmin").text) * scale_w),
                                      int(int(bbox.find("ymin").text) * scale_h),
                                      int(int(bbox.find("xmax").text) * scale_w),
                                      int(int(bbox.find("ymax").text) * scale_h)]
                objects.append(obj_struct)

            for obj in objects:
                nodeobject = doc.createElement("object")
                nodename = doc.createElement("name")
                nodebndbox = doc.createElement("bndbox")
                nodexmin = doc.createElement("xmin")
                nodeymin = doc.createElement("ymin")
                nodexmax = doc.createElement("xmax")
                nodeymax = doc.createElement("ymax")
                nodename.appendChild(doc.createTextNode(obj["name"]))
                nodexmin.appendChild(doc.createTextNode(str(obj["bbox"][0])))
                nodeymin.appendChild(doc.createTextNode(str(obj["bbox"][1])))
                nodexmax.appendChild(doc.createTextNode(str(obj["bbox"][2])))
                nodeymax.appendChild(doc.createTextNode(str(obj["bbox"][3])))

                nodebndbox.appendChild(nodexmin)
                nodebndbox.appendChild(nodeymin)
                nodebndbox.appendChild(nodexmax)
                nodebndbox.appendChild(nodeymax)

                nodeobject.appendChild(nodename)
                nodeobject.appendChild(nodebndbox)

                root.appendChild(nodeobject)

            fp = open(save_xml + imgname[:-8] + '_' + ('%d' % a) + '.xml', "w")
            doc.writexml(fp, indent='\t', addindent='\t',
                         newl='\n', encoding='utf-8')
            fp.close()

        else:
            # 接下来要对标签进行读取与处理
            pass
            '''
            xmlfile1 = xml_path + imgname[:-4] + '.xml'
            print("xml_1:", xmlfile1)
            # 读取两张图片的xml
            tree1 = ET.parse(xmlfile1)

            doc = xml.dom.minidom.Document()
            root = doc.createElement("annotation")
            doc.appendChild(root)

            nodeframe = doc.createElement("frame")
            nodeframe.appendChild(doc.createTextNode(imgname[:-8] + ('%d' % a)))

            objects = []
            for obj in tree1.findall("object"):
                obj_struct = {}
                obj_struct["name"] = obj.find("name").text
                bbox = obj.find("bndbox")
                obj_struct["bbox"] = [int(bbox.find("xmin").text),
                                      int(bbox.find("ymin").text),
                                      int(bbox.find("xmax").text),
                                      int(bbox.find("ymax").text)]
                objects.append(obj_struct)

            for obj in objects:
                nodeobject = doc.createElement("object")
                nodename = doc.createElement("name")
                nodebndbox = doc.createElement("bndbox")
                nodexmin = doc.createElement("xmin")
                nodeymin = doc.createElement("ymin")
                nodexmax = doc.createElement("xmax")
                nodeymax = doc.createElement("ymax")
                nodename.appendChild(doc.createTextNode(obj["name"]))
                nodexmin.appendChild(doc.createTextNode(str(obj["bbox"][0])))
                nodeymin.appendChild(doc.createTextNode(str(obj["bbox"][1])))
                nodexmax.appendChild(doc.createTextNode(str(obj["bbox"][2])))
                nodeymax.appendChild(doc.createTextNode(str(obj["bbox"][3])))

                nodebndbox.appendChild(nodexmin)
                nodebndbox.appendChild(nodeymin)
                nodebndbox.appendChild(nodexmax)
                nodebndbox.appendChild(nodeymax)

                nodeobject.appendChild(nodename)
                nodeobject.appendChild(nodebndbox)

                root.appendChild(nodeobject)

            fp = open(save_xml + imgname[:-8] + '_' + ('%d' % a) + '.xml', "w")
            doc.writexml(fp, indent='\t', addindent='\t',
                         newl='\n', encoding='utf-8')
            fp.close()
            '''

def rand_bbox(size, lamb):
    w = size[0]
    h = size[1]
    cut_rat = np.sqrt(1. - lamb)
    cut_w = np.int(w * cut_rat)
    cut_h = np.int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    return bbx1, bbx2, bby1, bby2

def generate_cutmix_image(image_batch, image_batch_labels, beta):
    lam = np.random.beta(beta, beta)
    rand_index = np.random.permutation(len(image_batch))
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch[0].shape, lam)
    print("bbx1:", bbx1)
    print("bby1:", bby1)
    print("bbx2:", bbx2)
    print("bby2:", bby2)
    image_batch_updated = image_batch.copy()
    image_batch_updated = np.array(image_batch_updated)
    image_batch = np.array(image_batch)


# 计算两个(xmin, ymin, xmax, ymax)的IOU
def bbox_iou(box1, box2, GIoU=False, DIoU=False, CIoU=False):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # print("value1:", max(0, (min(b1_x2, b2_x2) - max(b1_x1, b2_x1))))
    # print("value2:", max(0, (min(b1_y2, b2_y2) - max(b1_y1, b2_y1))))

    inter = max(0, (min(b1_x2, b2_x2) - max(b1_x1, b2_x1))) * \
            max(0, (min(b1_y2, b2_y2) - max(b1_y1, b2_y1)))
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter
    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


'''
获取同批次另一张图片的bbox目标，同时需要满足那个bbox目标和这张图片的iou不能太大
'''
def getbox(xmlfile1, xmlfile2):
    tree1 = ET.parse(xmlfile1)
    tree2 = ET.parse(xmlfile2)
    bbox1_return = []
    size1_return = []
    bbox2_return = []
    bbox2_name_return = []

    for obj in tree1.findall("object"):
        bbox1_temp = []
        bbox = obj.find("bndbox")
        bbox1_temp.append(int(bbox.find("xmin").text))
        bbox1_temp.append(int(bbox.find("ymin").text))
        bbox1_temp.append(int(bbox.find("xmax").text))
        bbox1_temp.append(int(bbox.find("ymax").text))
        bbox1_return.append(bbox1_temp)

    for size in tree1.findall("size"):
        size1_return.append(int(size.find("width").text))
        size1_return.append(int(size.find("height").text))
        size1_return.append(int(size.find("depth").text))


    for obj in tree2.findall("object"):
        bbox2_temp = []
        bbox = obj.find("bndbox")
        bbox2_temp.append(int(bbox.find("xmin").text))
        bbox2_temp.append(int(bbox.find("ymin").text))
        bbox2_temp.append(int(bbox.find("xmax").text))
        bbox2_temp.append(int(bbox.find("ymax").text))
        bbox2_return.append(bbox2_temp)
        bbox2_name_return.append(obj.find("name").text)
    # 同时也要把2的名字返回

    return size1_return, bbox1_return, bbox2_return, bbox2_name_return

def cutmix():
    a = 0
    for imgname in img_names:
        imgpath = img_path + imgname
        img = cv2.imread(imgpath)
        img_h, img_w = img.shape[0], img.shape[1]
        i = random.randint(0, img_num - 1)
        # i = 0

        add_path = img_path + img_names[i]
        addimg = cv2.imread(add_path)
        add_h, add_w = addimg.shape[0], addimg.shape[1]

        if add_h != img_h or add_w != img_w:
            print("resize!")
            addimg = cv2.resize(addimg, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        scale_h, scale_w = img_h / add_h, img_w / add_w
        print("scale_h:", scale_h)
        print("scale_w:", scale_w)
        # lamb = 0.5
        # size = img.shape
        # bbx1, bby1, bbx2, bby2 = rand_bbox(size, lamb)//变成获取addimg的目标位置
        xmlfile1 = xml_path + imgname[:-4] + '.xml'
        xmlfile2 = xml_path + img_names[i][:-4] + '.xml'
        # bbx1, bby1, bbx2, bby2 = getbox(xmlfile1, xmlfile2)
        size1, bbox1_list, bbox2_list, bbox2_name = getbox(xmlfile1, xmlfile2)
        len_bbox1 = len(bbox1_list)
        len_bbox2 = len(bbox2_list)
        if imgname != img_names[i]:  # 待增强的和加上去的那张图片不是同一张图片
            # 接下来要对标签进行读取与处理
            xmlfile1 = xml_path + imgname[:-4] + '.xml'
            print("xml_1_here:", xmlfile1)
            tree1 = ET.parse(xmlfile1)
            objects = []
            for obj in tree1.findall("object"):
                obj_struct = {}
                obj_struct["name"] = obj.find("name").text
                bbox = obj.find("bndbox")
                obj_struct["bbox"] = [int(bbox.find("xmin").text),
                                      int(bbox.find("ymin").text),
                                      int(bbox.find("xmax").text),
                                      int(bbox.find("ymax").text)]
                objects.append(obj_struct)  # 读取最原始的待修改图片的所有bbox标签变成list

            # 接下来开始创造新的xml
            doc = xml.dom.minidom.Document()
            root = doc.createElement("annotation")
            doc.appendChild(root)

            nodeframe = doc.createElement("frame")
            nodeframe.appendChild(doc.createTextNode(imgname[:-8] + '_' + ('%d' % a) + '.jpg'))
            root.appendChild(nodeframe)

            nodesize = doc.createElement("size")
            nodewidth = doc.createElement("width")
            nodewidth.appendChild(doc.createTextNode(str(size1[0])))
            nodeheight = doc.createElement("height")
            nodeheight.appendChild(doc.createTextNode(str(size1[1])))
            nodedepth = doc.createElement("depth")
            nodedepth.appendChild(doc.createTextNode(str(size1[2])))

            nodesize.appendChild(nodewidth)
            nodesize.appendChild(nodeheight)
            nodesize.appendChild(nodedepth)

            root.appendChild(nodesize)

            # 接下来要分别计算IOU
            temp_img = img.copy()
            for i_1 in range(len_bbox2):
                for j_1 in range(len_bbox1):
                    # 必须满足2的框要和1的所有框的IOU都满足条件才能赋值
                    bbox1 = bbox1_list[j_1]
                    bbox2 = bbox2_list[i_1]
                    box_iouvalue_test = bbox_iou(bbox1, bbox2)
                    if (box_iouvalue_test <= 0):
                        temp_img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]] = addimg[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
                        # 这里要把对应的坐标也添加进xml文件里面
                        obj_struct = {}
                        obj_struct["name"] = bbox2_name[i_1]
                        obj_struct["bbox"] = [int(int(bbox2[0]) * scale_w),
                                              int(int(bbox2[1]) * scale_h),
                                              int(int(bbox2[2]) * scale_w),
                                              int(int(bbox2[3]) * scale_h)]
                        objects.append(obj_struct)  # 只有符合IOU要求的坐标才会被写入

            # 把所有的objects写入
            for obj in objects:
                nodeobject = doc.createElement("object")
                nodename = doc.createElement("name")
                nodebndbox = doc.createElement("bndbox")
                nodexmin = doc.createElement("xmin")
                nodeymin = doc.createElement("ymin")
                nodexmax = doc.createElement("xmax")
                nodeymax = doc.createElement("ymax")
                nodename.appendChild(doc.createTextNode(obj["name"]))
                nodexmin.appendChild(doc.createTextNode(str(obj["bbox"][0])))
                nodeymin.appendChild(doc.createTextNode(str(obj["bbox"][1])))
                nodexmax.appendChild(doc.createTextNode(str(obj["bbox"][2])))
                nodeymax.appendChild(doc.createTextNode(str(obj["bbox"][3])))

                nodebndbox.appendChild(nodexmin)
                nodebndbox.appendChild(nodeymin)
                nodebndbox.appendChild(nodexmax)
                nodebndbox.appendChild(nodeymax)

                nodeobject.appendChild(nodename)
                nodeobject.appendChild(nodebndbox)

                root.appendChild(nodeobject)

            fp = open(save_xml + imgname[:-8] + '_' + ('%d' % a) + '.xml', "w")
            doc.writexml(fp, indent='\t', addindent='\t',
                         newl='\n', encoding='utf-8')
            fp.close()

            # 把图片写入保存
            save_img = save_path + imgname[:-8] + '_' + ('%d' % a) + '.jpg'
            cv2.imwrite(save_img, temp_img)
            print("save_img:", save_img)

        a = a + 1

        # 下面这些是单张测试的代码，不需要删除掉
        '''
        if a == 1:
            print('i:', i)
            print("xml_1:", xmlfile1)
            print("xml_2:", xmlfile2)
            print("list_1:", len(bbox1_list))
            print("list_2:", len(bbox2_list))
            print("bbox_list1:", bbox1_list)
            print("bbox_list2:", bbox2_list)
            print("size:", size1)
            #这次不一样，这次先创建xml文件，然后再往里面添加标签

            if imgname != img_names[i]:  # 待增强的和加上去的那张图片不是同一张图片
                # 接下来要对标签进行读取与处理
                xmlfile1 = xml_path + imgname[:-4] + '.xml'
                print("xml_1_here:", xmlfile1)
                tree1 = ET.parse(xmlfile1)
                objects = []
                for obj in tree1.findall("object"):
                    obj_struct = {}
                    obj_struct["name"] = obj.find("name").text
                    bbox = obj.find("bndbox")
                    obj_struct["bbox"] = [int(bbox.find("xmin").text),
                                          int(bbox.find("ymin").text),
                                          int(bbox.find("xmax").text),
                                          int(bbox.find("ymax").text)]
                    objects.append(obj_struct)#读取最原始的待修改图片的所有bbox标签变成list

                #接下来开始创造新的xml
                doc = xml.dom.minidom.Document()
                root = doc.createElement("annotation")
                doc.appendChild(root)

                nodeframe = doc.createElement("frame")
                nodeframe.appendChild(doc.createTextNode(imgname[:-8] + '_' + ('%d' % a) + '.jpg'))
                root.appendChild(nodeframe)

                nodesize = doc.createElement("size")
                nodewidth = doc.createElement("width")
                nodewidth.appendChild(doc.createTextNode(str(size1[0])))
                nodeheight = doc.createElement("height")
                nodeheight.appendChild(doc.createTextNode(str(size1[1])))
                nodedepth = doc.createElement("depth")
                nodedepth.appendChild(doc.createTextNode(str(size1[2])))

                nodesize.appendChild(nodewidth)
                nodesize.appendChild(nodeheight)
                nodesize.appendChild(nodedepth)

                root.appendChild(nodesize)

            #接下来要分别计算IOU
            temp_img = img.copy()
            for i_1 in range(len_bbox2):
                for j_1 in range(len_bbox1):
                    #必须满足2的框要和1的所有框的IOU都满足条件才能赋值
                    bbox1 = bbox1_list[j_1]
                    bbox2 = bbox2_list[i_1]
                    box_iouvalue_test = bbox_iou(bbox1, bbox2)
                    if(box_iouvalue_test <= 0):
                        temp_img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]] = addimg[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
                        #这里要把对应的坐标也添加进xml文件里面
                        obj_struct = {}
                        obj_struct["name"] = bbox2_name[i_1]
                        obj_struct["bbox"] = [int(int(bbox2[0])),
                                              int(int(bbox2[1])),
                                              int(int(bbox2[2])),
                                              int(int(bbox2[3]))]
                        objects.append(obj_struct)#只有符合IOU要求的坐标才会被写入

            #把所有的objects写入
            for obj in objects:
                nodeobject = doc.createElement("object")
                nodename = doc.createElement("name")
                nodebndbox = doc.createElement("bndbox")
                nodexmin = doc.createElement("xmin")
                nodeymin = doc.createElement("ymin")
                nodexmax = doc.createElement("xmax")
                nodeymax = doc.createElement("ymax")
                nodename.appendChild(doc.createTextNode(obj["name"]))
                nodexmin.appendChild(doc.createTextNode(str(obj["bbox"][0])))
                nodeymin.appendChild(doc.createTextNode(str(obj["bbox"][1])))
                nodexmax.appendChild(doc.createTextNode(str(obj["bbox"][2])))
                nodeymax.appendChild(doc.createTextNode(str(obj["bbox"][3])))

                nodebndbox.appendChild(nodexmin)
                nodebndbox.appendChild(nodeymin)
                nodebndbox.appendChild(nodexmax)
                nodebndbox.appendChild(nodeymax)

                nodeobject.appendChild(nodename)
                nodeobject.appendChild(nodebndbox)

                root.appendChild(nodeobject)

            fp = open(save_xml + imgname[:-8] + '_' + ('%d' % a) + '.xml', "w")
            doc.writexml(fp, indent='\t', addindent='\t',
                         newl='\n', encoding='utf-8')
            fp.close()

            #把图片写入保存
            save_img = save_path + imgname[:-8] + '_' + ('%d' % a) + '.jpg'
            cv2.imwrite(save_img, temp_img)
            print("save_img:", save_img)
        '''
        '''
        bbox1 = bbox1_list[0]
        bbox2 = bbox2_list[0]

        box_iouvalue_test = bbox_iou(bbox1, bbox2)#计算出两个目标的iou,必须要iou<=0的时候才能进行扩增
        print("box_iouvalue_test:", box_iouvalue_test)

        if box_iouvalue_test <= 0:
            temp_img = img.copy()
            temp_img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]] = addimg[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
            save_img = save_path + imgname[:-8] + '_' + ('%d' % a) + '.jpg'
            cv2.imwrite(save_img, temp_img)
            print("save_img:", save_img)
        '''
        '''
        if a == 0:
            if imgname != img_names[i]:  # 待增强的和加上去的那张图片不是同一张图片
                # 接下来要对标签进行读取与处理
                xmlfile1 = xml_path + imgname[:-4] + '.xml'
                xmlfile2 = xml_path + img_names[i][:-4] + '.xml'
                print("xml_1:", xmlfile1)
                print("xml_2:", xmlfile2)
                # 读取两张图片的xml
                tree1 = ET.parse(xmlfile1)
                tree2 = ET.parse(xmlfile2)
                doc = xml.dom.minidom.Document()
                root = doc.createElement("annotation")
                doc.appendChild(root)

                nodeframe = doc.createElement("frame")
                nodeframe.appendChild(doc.createTextNode(imgname[:-8] + ('%d' % a)))

                objects = []
                for obj in tree1.findall("object"):
                    obj_struct = {}
                    obj_struct["name"] = obj.find("name").text
                    bbox = obj.find("bndbox")
                    obj_struct["bbox"] = [int(bbox.find("xmin").text),
                                          int(bbox.find("ymin").text),
                                          int(bbox.find("xmax").text),
                                          int(bbox.find("ymax").text)]
                    objects.append(obj_struct)


                #把这一段改成读取通过验证的bbox
                for obj in tree2.findall("object"):
                    obj_struct = {}
                    obj_struct["name"] = obj.find("name").text
                    bbox = obj.find("bndbox")
                    obj_struct["bbox"] = [int(int(bbox.find("xmin").text) * scale_w),
                                          int(int(bbox.find("ymin").text) * scale_h),
                                          int(int(bbox.find("xmax").text) * scale_w),
                                          int(int(bbox.find("ymax").text) * scale_h)]
                    objects.append(obj_struct)

                for obj in objects:
                    nodeobject = doc.createElement("object")
                    nodename = doc.createElement("name")
                    nodebndbox = doc.createElement("bndbox")
                    nodexmin = doc.createElement("xmin")
                    nodeymin = doc.createElement("ymin")
                    nodexmax = doc.createElement("xmax")
                    nodeymax = doc.createElement("ymax")
                    nodename.appendChild(doc.createTextNode(obj["name"]))
                    nodexmin.appendChild(doc.createTextNode(str(obj["bbox"][0])))
                    nodeymin.appendChild(doc.createTextNode(str(obj["bbox"][1])))
                    nodexmax.appendChild(doc.createTextNode(str(obj["bbox"][2])))
                    nodeymax.appendChild(doc.createTextNode(str(obj["bbox"][3])))

                    nodebndbox.appendChild(nodexmin)
                    nodebndbox.appendChild(nodeymin)
                    nodebndbox.appendChild(nodexmax)
                    nodebndbox.appendChild(nodeymax)

                    nodeobject.appendChild(nodename)
                    nodeobject.appendChild(nodebndbox)

                    root.appendChild(nodeobject)

                fp = open(save_xml + imgname[:-8] + '_' + ('%d' % a) + '.xml', "w")
                doc.writexml(fp, indent='\t', addindent='\t',
                             newl='\n', encoding='utf-8')
                fp.close()
        '''

    # print("size:", size)(360, 640, 3)

