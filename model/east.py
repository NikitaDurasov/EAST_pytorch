import torch
import torchvision


class EAST(torch.nn.Module):

    # TODO different kinds of backbones
    @staticmethod
    def build_feature_extractor():
        resnet_params = torchvision.models.resnet50(pretrained=True)

        # 3 -> 64, /2
        block_1 = torch.nn.Sequential(resnet_params.conv1,
                                      resnet_params.bn1,
                                      resnet_params.relu)

        # 64 -> 256, /2
        block_2 = torch.nn.Sequential(resnet_params.maxpool,
                                      resnet_params.layer1)

        # 256 -> 512, /2
        block_3 = resnet_params.layer2

        # 512 -> 1024, /2
        block_4 = resnet_params.layer3

        # 1024 -> 2048, /2
        block_5 = resnet_params.layer4

        return torch.nn.Sequential(block_1, block_2, block_3, block_4, block_5)

    @staticmethod
    def build_feature_merging_branch():
        # 2048 -> 1024, 2/
        block_1 = torch.nn.Sequential(torch.nn.Conv2d(2048, 1024, 1),
                                      torch.nn.Conv2d(1024, 1024, 3, padding=1))

        # 1024 -> 512, 2/
        block_2 = torch.nn.Sequential(torch.nn.Conv2d(2*1024, 512, 1),
                                      torch.nn.Conv2d(512, 512, 3, padding=1))

        # 512 -> 256, 2/
        block_3 = torch.nn.Sequential(torch.nn.Conv2d(2*512, 256, 1),
                                      torch.nn.Conv2d(256, 256, 3, padding=1))

        # 256 -> 64, 2/
        block_4 = torch.nn.Sequential(torch.nn.Conv2d(2*256, 64, 1),
                                      torch.nn.Conv2d(64, 64, 3, padding=1))

        # 64 -> 1, 2/
        block_5 = torch.nn.Conv2d(2*64, 32, 3, padding=1)

        return torch.nn.Sequential(block_1, block_2, block_3, block_4, block_5)

    def __init__(self, kind="quad"):
        super(EAST, self).__init__()
        self.resnet_blocks = self.build_feature_extractor()
        self.feature_merging_branch = self.build_feature_merging_branch()
        self.score_map_conv = torch.nn.Conv2d(32, 1, 1)
        self.quad_conv = torch.nn.Conv2d(32, 8, 1)
        self.kind = kind

    def forward(self, x):

        # interpolation block
        unpooling = torch.nn.Upsample(scale_factor=2,
                                      mode='bilinear',
                                      align_corners=True)

        # feature extraction
        fte1 = self.resnet_blocks[0](x)
        fte2 = self.resnet_blocks[1](fte1)
        fte3 = self.resnet_blocks[2](fte2)
        fte4 = self.resnet_blocks[3](fte3)
        fte5 = self.resnet_blocks[4](fte4)

        # features merging
        ftm5 = fte5
        ftm5 = unpooling(ftm5)

        ftm4 = torch.cat([self.feature_merging_branch[0](ftm5), fte4], dim=1)
        ftm4 = unpooling(ftm4)

        ftm3 = torch.cat([self.feature_merging_branch[1](ftm4), fte3], dim=1)
        ftm3 = unpooling(ftm3)

        ftm2 = torch.cat([self.feature_merging_branch[2](ftm3), fte2], dim=1)
        ftm2 = unpooling(ftm2)

        ftm1 = torch.cat([self.feature_merging_branch[3](ftm2), fte1], dim=1)
        ftm1 = unpooling(ftm1)

        merged_features = self.feature_merging_branch[4](ftm1)

        score_map = torch.nn.functional.sigmoid(self.score_map_conv(merged_features))
        if self.kind == 'quad':
            geometry = self.quad_conv(merged_features)

        return {'score_map': score_map,
                'geometry': geometry}
