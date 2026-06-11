# UNet-HandWritten

U-Net- [This is not the vanilla one , we added residual blocks into it]

This is important CV model which is used for semantic segmentation i.e if 5 cars are there it will give label as car for each of the car
Where as instance segmentation (achieved by Masked R-CNN) i.e if 5 cards are there then it will give label as car1 ,car2, car3... i.e each object different label


Important Parts of the UNet 
1.Residual Blocks  [In vanilla U-Net , Ig they didn't use the idea of Residual Blocks , But using Residual Blocks instead of DoubleConv Blocks]
2.Skip Connections



BELOW IS THE U-Net ARCHITECTURE






 [ INPUT IMAGE ]                                                      [ OUTPUT MASK ]
  (256 x 256 x 3)                                                      (256 x 256 x 1)
        │                                                                     ▲
        ▼                                                                     │
 ┌──────────────┐                                                      ┌──────────────┐
 │ Encoder 1    │ - - - - - - - [ SKIP CONNECTION ] - - - - - - - - - >│ Decoder 1    │
 │ 256x256x64   │                                                      │ 256x256x64   │
 └──────────────┘                                                      └──────────────┘
        │                                                                     ▲
    Max Pool 2x2                                                         Up-Conv 2x2
        ▼                                                                     │
 ┌──────────────┐                                                      ┌──────────────┐
 │ Encoder 2    │ - - - - - - - [ SKIP CONNECTION ] - - - - - - - - - >│ Decoder 2    │
 │ 128x128x128  │                                                      │ 128x128x128  │
 └──────────────┘                                                      └──────────────┘
        │                                                                     ▲
    Max Pool 2x2                                                         Up-Conv 2x2
        ▼                                                                     │
 ┌──────────────┐                                                      ┌──────────────┐
 │ Encoder 3    │ - - - - - - - [ SKIP CONNECTION ] - - - - - - - - - >│ Decoder 3    │
 │  64x64x256   │                                                      │  64x64x256   │
 └──────────────┘                                                      └──────────────┘
        │                                                                     ▲
    Max Pool 2x2                                                         Up-Conv 2x2
        ▼                                                                     │
 ┌──────────────┐                                                      ┌──────────────┐
 │ Encoder 4    │ - - - - - - - [ SKIP CONNECTION ] - - - - - - - - - >│ Decoder 4    │  [Inside Up-Conv(16x16x1024) => 32x32x512 concate with the Encoder 32x32x512
 │  32x32x512   │                                                      │  32x32x512   │		which is 32x32x1024 => 1x1 convlutions to change the channels 1024=>512
 └──────────────┘                                                      └──────────────┘
        │                                                                     ▲
    Max Pool 2x2                                                         Up-Conv 2x2
        │                                                                     │
        └───────────────────────────────> ┌──────────────┐ ───────────────────┘
                                          │ BOTTLENECK   │
                                          │  16x16x1024  │
                                          └──────────────┘














THE ENCODER ARCHITECTURE

               Tensor Input from Previous Layer 
                     (e.g., 256 x 256 x 3)
                              │
             ┌────────────────┴────────────────┐
             ▼ (Main Path)                     ▼ (Shortcut Path)
     ┌───────────────┐                 ┌───────────────┐
     │  Conv 3x3     │                 │  Conv 1x1     │ (Adjusts channels
     │ (Channels: 64)│                 │ (Channels: 64)│  if input != output)
     └───────┬───────┘                 └───────┬───────┘
             ▼                                 │
     ┌───────────────┐                         │
     │ BatchNorm +   │                         │
     │ ReLU Activation                         │
     └───────┬───────┘                         │
             ▼                                 │
     ┌───────────────┐                         │
     │  Conv 3x3     │                         │
     │ (Channels: 64)│                         │
     └───────┬───────┘                         │
             ▼                                 │
     ┌───────────────┐                         │
     │  BatchNorm    │                         │
     └───────┬───────┘                         │
             │                                 │
             └───────────────> ⊕ <─────────────┘
                        Element-wise Plus
                              │
                              ▼
                       ┌───────────────┐
                       │     ReLU      │
                       └───────┬───────┘
                               │
             ┌─────────────────┴─────────────────┐
             │                                   │
             ▼                                   ▼
    [ SEND TO GLOBAL SKIP ]               [ SEND TO DOWN-SAMPLING ]
   (Goes across the "U" to             (Max Pool or Strided Conv 
      Decoder Block 1)                    to change size to 128x128)












DECODER ARCHITECTURE



    Tensor From Lower Layer               Tensor From Encoder
        (128 x 128 x 128)                    (256 x 256 x 64)
               │                                    │
               ▼                                    │
       ┌───────────────┐                            │
       │  Up-Conv 2x2  │                            │
       │ (256x256x64)  │                            │
       └───────┬───────┘                            │
               │                                    │
               └───────────────> █ <────────────────┘
                          Concatenate Along
                           Channel Axis
                                │
                                ▼
                      Combined Tensor Output
                         (256 x 256 x 128)
                                │
             ┌──────────────────┴──────────────────┐
             ▼ (Main Path)                         ▼ (Shortcut Path)
     ┌───────────────┐                     ┌───────────────┐
     │  Conv 3x3     │                     │  Conv 1x1     │ (Reduces channels 
     │ (Channels: 64)│                     │ (Channels: 64)│  back to target)
     └───────┬───────┘                     └───────┬───────┘
             ▼                                     │
     ┌───────────────┐                             │
     │ BatchNorm +   │                             │
     │ ReLU Activation                             │
     └───────┬───────┘                             │
             ▼                                     │
     ┌───────────────┐                             │
     │  Conv 3x3     │                             │
     │ (Channels: 64)│                             │
     └───────┬───────┘                             │
             ▼                                     │
     ┌───────────────┐                             │
     │  BatchNorm    │                             │
     └───────┬───────┘                             │
             │                                     │
             └────────────────> ⊕ <────────────────┘
                        Element-wise Plus
                              │
                              ▼
                       ┌───────────────┐
                       │     ReLU      │
                       └───────┬───────┘
                               │
                               ▼
                    Output to Next Decoder Level 
                         (256 x 256 x 64)
