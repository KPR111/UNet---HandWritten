# U-Net- [This is not the vanilla one , we added residual blocks into it]

# This is important CV model which is used for semantic segmentation i.e if 5 cars are there it will give label as car for each of the car
# Where as instance segmentation (achieved by Masked R-CNN) i.e if 5 cards are there then it will give label as car1 ,car2, car3... i.e each object different label


# Important Parts of the UNet 
# 1.Residual Blocks  [In vanilla U-Net , Ig they didn't use the idea of Residual Blocks , But using Residual Blocks instead of DoubleConv Blocks]
# 2.Skip Connections



# BELOW IS THE U-Net ARCHITECTURE





#  [ INPUT IMAGE ]                                                      [ OUTPUT MASK ]
#   (256 x 256 x 3)                                                      (256 x 256 x 1)
#         │                                                                     ▲
#         ▼                                                                     │
#  ┌──────────────┐                                                      ┌──────────────┐
#  │ Encoder 1    │ - - - - - - - [ SKIP CONNECTION ] - - - - - - - - - >│ Decoder 1    │
#  │ 256x256x64   │                                                      │ 256x256x64   │
#  └──────────────┘                                                      └──────────────┘
#         │                                                                     ▲
#     Max Pool 2x2                                                         Up-Conv 2x2
#         ▼                                                                     │
#  ┌──────────────┐                                                      ┌──────────────┐
#  │ Encoder 2    │ - - - - - - - [ SKIP CONNECTION ] - - - - - - - - - >│ Decoder 2    │
#  │ 128x128x128  │                                                      │ 128x128x128  │
#  └──────────────┘                                                      └──────────────┘
#         │                                                                     ▲
#     Max Pool 2x2                                                         Up-Conv 2x2
#         ▼                                                                     │
#  ┌──────────────┐                                                      ┌──────────────┐
#  │ Encoder 3    │ - - - - - - - [ SKIP CONNECTION ] - - - - - - - - - >│ Decoder 3    │
#  │  64x64x256   │                                                      │  64x64x256   │
#  └──────────────┘                                                      └──────────────┘
#         │                                                                     ▲
#     Max Pool 2x2                                                         Up-Conv 2x2
#         ▼                                                                     │
#  ┌──────────────┐                                                      ┌──────────────┐
#  │ Encoder 4    │ - - - - - - - [ SKIP CONNECTION ] - - - - - - - - - >│ Decoder 4    │  [Inside Up-Conv(16x16x1024) => 32x32x512 concate with the Encoder 32x32x512
#  │  32x32x512   │                                                      │  32x32x512   │		which is 32x32x1024 => 1x1 convlutions to change the channels 1024=>512
#  └──────────────┘                                                      └──────────────┘
#         │                                                                     ▲
#     Max Pool 2x2                                                         Up-Conv 2x2
#         │                                                                     │
#         └───────────────────────────────> ┌──────────────┐ ───────────────────┘
#                                           │ BOTTLENECK   │
#                                           │  16x16x1024  │
#                                           └──────────────┘














# THE ENCODER ARCHITECTURE

#                Tensor Input from Previous Layer 
#                      (e.g., 256 x 256 x 3)
#                               │
#              ┌────────────────┴────────────────┐
#              ▼ (Main Path)                     ▼ (Shortcut Path)
#      ┌───────────────┐                 ┌───────────────┐
#      │  Conv 3x3     │                 │  Conv 1x1     │ (Adjusts channels
#      │ (Channels: 64)│                 │ (Channels: 64)│  if input != output)
#      └───────┬───────┘                 └───────┬───────┘
#              ▼                                 │
#      ┌───────────────┐                         │
#      │ BatchNorm +   │                         │
#      │ ReLU Activation                         │
#      └───────┬───────┘                         │
#              ▼                                 │
#      ┌───────────────┐                         │
#      │  Conv 3x3     │                         │
#      │ (Channels: 64)│                         │
#      └───────┬───────┘                         │
#              ▼                                 │
#      ┌───────────────┐                         │
#      │  BatchNorm    │                         │
#      └───────┬───────┘                         │
#              │                                 │
#              └───────────────> ⊕ <─────────────┘
#                         Element-wise Plus
#                               │
#                               ▼
#                        ┌───────────────┐
#                        │     ReLU      │
#                        └───────┬───────┘
#                                │
#              ┌─────────────────┴─────────────────┐
#              │                                   │
#              ▼                                   ▼
#     [ SEND TO GLOBAL SKIP ]               [ SEND TO DOWN-SAMPLING ]
#    (Goes across the "U" to             (Max Pool or Strided Conv 
#       Decoder Block 1)                    to change size to 128x128)












# DECODER ARCHITECTURE



#     Tensor From Lower Layer               Tensor From Encoder
#         (128 x 128 x 128)                    (256 x 256 x 64)
#                │                                    │
#                ▼                                    │
#        ┌───────────────┐                            │
#        │  Up-Conv 2x2  │                            │
#        │ (256x256x64)  │                            │
#        └───────┬───────┘                            │
#                │                                    │
#                └───────────────> █ <────────────────┘
#                           Concatenate Along
#                            Channel Axis
#                                 │
#                                 ▼
#                       Combined Tensor Output
#                          (256 x 256 x 128)
#                                 │
#              ┌──────────────────┴──────────────────┐
#              ▼ (Main Path)                         ▼ (Shortcut Path)
#      ┌───────────────┐                     ┌───────────────┐
#      │  Conv 3x3     │                     │  Conv 1x1     │ (Reduces channels 
#      │ (Channels: 64)│                     │ (Channels: 64)│  back to target)
#      └───────┬───────┘                     └───────┬───────┘
#              ▼                                     │
#      ┌───────────────┐                             │
#      │ BatchNorm +   │                             │
#      │ ReLU Activation                             │
#      └───────┬───────┘                             │
#              ▼                                     │
#      ┌───────────────┐                             │
#      │  Conv 3x3     │                             │
#      │ (Channels: 64)│                             │
#      └───────┬───────┘                             │
#              ▼                                     │
#      ┌───────────────┐                             │
#      │  BatchNorm    │                             │
#      └───────┬───────┘                             │
#              │                                     │
#              └────────────────> ⊕ <────────────────┘
#                         Element-wise Plus
#                               │
#                               ▼
#                        ┌───────────────┐
#                        │     ReLU      │
#                        └───────┬───────┘
#                                │
#                                ▼
#                     Output to Next Decoder Level 
#                          (256 x 256 x 64)




clas UNetConfig():
	H = 256
	W = 256
	C = 3
	
class ResidualBlock(nn.Module):
	def __init__(self,in_channels,out_channels):
		super().__init__()
		self.config = config
		self.conv1 = nn.Conv2d(in_channels,out_channels,kernel = 3 , stride = 1 , padding = 1)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace = True)
		self.conv2 = nn.Conv2d(out_channels,out_channels,kernel = 3 , stride =1 , padding = 1)
		self.bn2 = nn.BatchNorm2d(out_channels)
		
		if(in_channels!=out_channels):
			self.shortcut = nn.Conv2d(in_channels,out_channels,kernel=1,stride=1)
		else 
			self.shortcut = nn.Identity()
			
		
	def forward(self,x):
		y = self.shortcut(x)
		
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		
		x = self.conv2(x)
		x = self.bn2(x)
		
		x = x + y
		x = self.relu(x)
		
		return x
		
		
		

class DecoderStep(nn.Module):
	def __init__(self,in_channels,out_channels,skip_channels):
		super().__init__()
		self.up_conv = self.TransposeConv2d(in_channels,out_channels,kernel_size = 2,stride = 2)
		combined_channels = out_channels + skip_channels
		self.res_block = ResidualBlock(combined_channels,out_channels)
		
	def forward(self,x,skip_connections):
		x = sel.up_conv(x)
		x_combined = torch.cat([x,skip_connections],dim=1) #[B,C,H,W] dim =1 channel dimension
		
		return self.res_block(x_combined)
		
		
class EncoderStep(nn.Module):
	def __init__(self,in_channels,out_channels):
		super().__init__()
		self.pool = nn.MaxPool2d(kernel_size = 2 , stride = 2)
		self.res_block = ResidualBlock(in_channels,out_channels)
		
	def forward(self,x):
		skip_connections = self.res_block(x)  #this one is used while decoder we will concat this, so that we can recover lost information
		down_sampled_features = self.pool(skip_connections)
		
		return down_sampled_features , skip_connections
		


class UNet(nn.Module):
	def __init__(self,config,num_classes):
		super().__init__()
		self.config = config
		
		self.enc1 = Encoder(3,64)
		self.enc2 = Encoder(64,128)
		self.enc3 = Encoder(128,256)
		self.enc4 = Encoder(256,512)
		
		#The bottom base - BOTTLENECK
		self.bottleneck = ResidualBlock(512,1024)
		
		self.dec4 = Decoder(1024,512,512)
		self.dec3 = Decoder(512,256,256)
		self.dec2 = Decoder(256,128,128)
		self.dec1 = Decoder(128,64,128)
		
		self.final_head = nn.Conv2d(64,num_classes,kernel_size=1)
		
		
	
	def forward(self,x):
	
		x , skip1 = self.enc1(x) #[B,64,256,256]
		x , skip2 = self.enc2(x) #[B,128,128,128]
		x , skip3 = self.enc3(x) #[B,256,64,64]
		x , skip4 = self.enc4(x) #[B,512,32,32]
		
		x = self.bottleneck(x)
		
		x = self.dec4(x,skip4)
		x = self.dec3(x,skip3)
		x = self.dec2(x,skip2)
		x = self.dec1(x,skip1)
		
		output_logits = self.final_head(x)
		return output_logits
