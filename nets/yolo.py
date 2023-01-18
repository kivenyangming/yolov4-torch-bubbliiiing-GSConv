from nets.CSPdarknet import *


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#
        self.backbone = CSPDarknet(base_channels, base_depth, phi, pretrained)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.concat = Concat(dimension=1)
        self.SPPF = SPPF(base_channels * 16, base_channels * 8)  # 1024 ---> 512
        self.P4Conv = Conv(base_channels * 8, base_channels * 4)  # 1,512,40,40 ---> 1,256,40,40
        self.P3Conv = Conv(base_channels * 4, base_channels * 2)  # 1,512,40,40 ---> 1,256,40,40

        self.P5GSConv = GSConv(base_channels * 8, base_channels * 4)  # 1,512,20,20 ---> 1,256,20,20

        self.P4VoV = VoVGSCSP(base_channels * 8, base_channels * 4)  # 1,512,40,40 ---> 1,256,40,40
        self.P4GSConv = GSConv(base_channels * 4, base_channels * 2)  # 1,256,40,40 ---> 1,128,40,40

        self.P3VoV = VoVGSCSP(base_channels * 4, base_channels * 2)  # 1,256,80,80 ---> 1,128,80,80
        self.P3GSConvH = GSConv(base_channels * 2, base_channels * 4)  # 1,128,80,80 ---> 1,256,80,80
        self.P3GSConv = GSConv(base_channels * 2, base_channels * 4, 3, 2)  # 1,128,80,80 ---> 1,256,40,40

        self.Head2VoV = VoVGSCSP(base_channels * 8, base_channels * 4)  # 1,512,40,40 ---> 1,256,40,40
        self.Head2GSConv = GSConv(base_channels * 4, base_channels * 8)  # 1,256,40,40 ---> 1,512,40,40

        self.Head3GSConv1 = GSConv(base_channels * 4, base_channels * 8, 3, 2)  # 1,256,20,20 ---> 1,512,20,20
        self.Head3VoV = VoVGSCSP(base_channels * 16, base_channels * 8)  # 1,1024,20,20 ---> 1,512,20,20
        self.Head3GSConv2 = GSConv(base_channels * 8, base_channels * 16)  # 1,512,20,20 ---> 1,1024,20,20

        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, x):
        P3, P4, P5 = self.backbone(x)

        P5SPPF = self.SPPF(P5)
        P5 = self.P5GSConv(P5SPPF)
        P5P5SPPF_Up = self.upsample(P5)

        P4 = self.P4Conv(P4)
        P4 = self.concat([P4, P5P5SPPF_Up])

        P4VoV = self.P4VoV(P4)
        P4 = self.P4GSConv(P4VoV)
        P4_Up = self.upsample(P4)

        P3 = self.P3Conv(P3)
        P3 = self.concat([P3, P4_Up])
        P3 = self.P3VoV(P3)
        Head1 = self.P3GSConvH(P3)
        P3G = self.P3GSConv(P3)
        P3C = self.concat([P3G, P4VoV])

        Head2VoV = self.Head2VoV(P3C)
        Head2 = self.Head2GSConv(Head2VoV)

        Head3G1 = self.Head3GSConv1(Head2VoV)
        Head3C = self.concat([Head3G1, P5SPPF])
        Head3V = self.Head3VoV(Head3C)
        Head3 = self.Head3GSConv2(Head3V)

        Out1 = self.yolo_head_P3(Head1)
        Out2 = self.yolo_head_P4(Head2)
        Out3 = self.yolo_head_P5(Head3)

        return Out3, Out2, Out1


if __name__ == "__main__":
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes = 80
    phi = 'l'
    model = YoloBody(anchors_mask, num_classes, phi, pretrained=False)
    x = torch.ones((1, 3, 640, 640))
    Out3, Out2, Out1 = model(x)
    print(Out3.shape, Out2.shape, Out1.shape)

