● Based on my analysis, here is the comprehensive architectural breakdown of the Watermark Freedom codebase:                                                          
                                                                                                                                                                      
  ---                                                                                                                                                                 
  Watermark Freedom - Architecture Analysis                                                                                                                           
                                                                                                                                                                      
  1. SYSTEM ARCHITECTURE OVERVIEW                                                                                                                                     
                                                                                                                                                                      
  ┌─────────────────────────────────────────────────────────────────────────────┐                                                                                     
  │                     WATERMARK FREEDOM SYSTEM                                │                                                                                     
  │              Localized Neural Watermark Embedding/Detection                 │                                                                                     
  └─────────────────────────────────────────────────────────────────────────────┘                                                                                     
                                        │                                                                                                                             
          ┌─────────────────────────────┴─────────────────────────────┐                                                                                               
          │                                                           │                                                                                               
          ▼                                                           ▼                                                                                               
  ┌───────────────────┐                                       ┌───────────────────┐                                                                                   
  │   EMBED MODE      │                                       │   VERIFY MODE     │                                                                                   
  │  (Watermarking)   │                                       │   (Detection)     │                                                                                   
  └───────────────────┘                                       └───────────────────┘                                                                                   
                                                                                                                                                                      
  ---                                                                                                                                                                 
  2. MODULE HIERARCHY                                                                                                                                                 
                                                                                                                                                                      
  watermark-freedom/                                              
  │                                                                                                                                                                   
  ├── core.py                          # WatermarkManager (Main orchestration)                                                                                        
  │   ├── embed()                      # Watermark embedding pipeline                                                                                                 
  │   └── verify()                     # Watermark verification pipeline                                                                                              
  │                                                                                                                                                                   
  ├── watermark_anything/              # Core WAM library (from Meta)                                                                                                 
  │   ├── models/                                                                                                                                                     
  │   │   ├── wam.py                  # WAM model (combines embedder+extractor)                                                                                       
  │   │   ├── embedder.py             # VAE-based watermark embedder                                                                                                  
  │   │   └── extractor.py            # SAM-based watermark detector                                                                                                  
  │   ├── modules/                                                                                                                                                    
  │   │   ├── vae.py                  # VAE encoder/decoder                                                                                                           
  │   │   ├── vit.py                  # Vision Transformer (SAM encoder)                                                                                              
  │   │   ├── pixel_decoder.py        # Pixel decoder for detection                                                                                                   
  │   │   └── msg_processor.py        # Message embedding processor                                                                                                   
  │   ├── augmentation/                                                                                                                                               
  │   │   └── augmenter.py            # Data augmentation for robustness                                                                                              
  │   └── data/                                                                                                                                                       
  │       └── transforms.py           # Image transforms                                                                                                              
  │                                                                                                                                                                   
  ├── viewframe.py                     # Corner bracket overlay system                                                                                                
  ├── viewframe_detector.py            # Viewframe corner detection                                                                                                   
  ├── roco_core.py                     # ROCO encoding (payload ↔ bits)                                                                                               
  ├── roco_ecc.py                      # ECC encoding/decoding                                                                                                        
  └── watermark_utils.py               # Utility functions                                                                                                            
                                                                                                                                                                      
  ---                                                                                                                                                                 
  3. COMPONENT RELATIONSHIP GRAPH                                                                                                                                     
                                                                                                                                                                      
                      ┌──────────────────────┐                    
                      │  WatermarkManager    │                                                                                                                        
                      │    (core.py)         │                                                                                                                        
                      └──────────┬───────────┘                                                                                                                        
                                 │                                                                                                                                    
          ┌──────────────────────┼──────────────────────┐                                                                                                             
          │                      │                      │                                                                                                             
          ▼                      ▼                      ▼                                                                                                             
  ┌─────────────┐       ┌───────────────┐      ┌──────────────┐                                                                                                       
  │    WAM      │       │ViewframeDetector│      │  ROCO ECC   │                                                                                                      
  │  (Model)    │       │  (Detection)   │      │  (Encoding)  │                                                                                                      
  └──────┬──────┘       └───────────────┘      └──────────────┘                                                                                                       
         │                                                                                                                                                            
         ├─────────────────────────────────────┐                                                                                                                      
         │                                     │                                                                                                                      
         ▼                                     ▼                                                                                                                      
  ┌──────────────┐                   ┌─────────────────┐                                                                                                              
  │  Embedder    │                   │   Extractor     │                                                                                                              
  │  (VAE-based) │                   │  (SAM-based)    │                                                                                                              
  └──────┬───────┘                   └────────┬────────┘                                                                                                              
         │                                    │                                                                                                                       
         ├──────────────┬─────────────────────┴──────────────┐                                                                                                        
         │              │                                     │                                                                                                       
         ▼              ▼                                     ▼                                                                                                       
  ┌──────────┐  ┌──────────────┐                    ┌────────────────┐                                                                                                
  │VAEEncoder│  │MsgProcessor │                    │ImageEncoderViT │                                                                                                 
  │          │  │             │                    │  (ViT Tiny)    │                                                                                                 
  └──────────┘  └──────┬──────┘                    └────────┬───────┘                                                                                                 
                       │                                    │                                                                                                         
                       ▼                                    ▼                                                                                                         
                ┌───────────┐                     ┌────────────────┐                                                                                                  
                │VAEDecoder │                     │  PixelDecoder  │                                                                                                  
                └───────────┘                     └────────────────┘                                                                                                  
                                                                                                                                                                      
  ---                                                                                                                                                                 
  4. DATA FLOW PIPELINES                                                                                                                                              
                                                                                                                                                                      
  4.1 EMBEDDING PIPELINE                                          
                                                                                                                                                                      
  Input: Image + Message (max 3 chars)                                                                                                                                
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 1. PREPROCESSING                                            │                                                                                                     
  │    └─ load_image() → crop_to_centered_square() →           │                                                                                                      
  │       default_transform() → Tensor[B=1, C=3, H, W]         │                                                                                                      
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 2. VIEWFRAME CALCULATION                                    │                                                                                                     
  │    └─ _get_viewframe_region() → (x, y, width, height)      │                                                                                                      
  │       Modes: "corners" (default), "pixels", "percentage"   │                                                                                                      
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 3. CROP TO VIEWFRAME                                        │                                                                                                     
  │    └─ img[:, :, y:y+h, x:x+w]                               │                                                                                                     
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 4. RESIZE TO 256×256 (WAM input size)                       │                                                                                                     
  │    └─ F.interpolate(size=(256,256), mode="bilinear")        │                                                                                                     
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 5. MESSAGE ENCODING                                         │                                                                                                     
  │    └─ roco_encode_to_binary_tensor() → Tensor[32 bits]      │                                                                                                     
  │       "ABC" → [1,0,1,0,0,...]                              │                                                                                                      
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 6. WAM.EMBED()                                              │                                                                                                     
  │    ├── VAE Encoder: Image → Latent                          │                                                                                                     
  │    ├── MsgProcessor: Latent + Message → Watermarked Latent  │                                                                                                     
  │    └── VAE Decoder: Watermarked Latent → Delta Image        │                                                                                                     
  │    └─ Blend: Original + scaling_w × Delta = Watermarked     │                                                                                                     
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 7. RESIZE BACK TO VIEWFRAME SIZE                            │                                                                                                     
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 8. COMPOSITE BACK INTO ORIGINAL IMAGE                       │                                                                                                     
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 9. DRAW CORNER BRACKETS                                     │                                                                                                     
  │    └─ draw_corner_brackets() with method:                   │                                                                                                     
  │       • "distinctive" (default): pixels 254/1              │                                                                                                      
  │       • "alpha": alpha-blended overlay                      │                                                                                                     
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
      Output: Watermarked Image Tensor                                                                                                                                
                                                                                                                                                                      
  4.2 VERIFICATION PIPELINE                                                                                                                                           
                                                                                                                                                                      
  Input: Watermarked Image                                                                                                                                            
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 1. PREPROCESSING (same as embed)                            │                                                                                                     
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 2. VIEWFRAME DETECTION                                      │                                                                                                     
  │    └─ _detect_viewframe() → ViewframeDetector.detect()      │                                                                                                     
  │       • Convert to grayscale                                │                                                                                                     
  │       • Threshold for pixels == 254 or == 1                │                                                                                                      
  │       • Morphological dilation                              │                                                                                                     
  │       • Find contours, locate 4 corners                    │                                                                                                      
  │       • Calculate bounding box                              │                                                                                                     
  │    └─ Fallback: centered square if detection fails          │                                                                                                     
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 3. CROP & RESIZE TO 256×256                                 │                                                                                                     
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 4. WAM.DETECT()                                             │                                                                                                     
  │    ├── ImageEncoderViT: Image → Features                   │                                                                                                      
  │    └── PixelDecoder: Features → Predictions                │                                                                                                      
  │    Output: Tensor[B=1, C=33, H=256, W=256]                 │                                                                                                      
  │           Channel 0: Mask prediction                        │                                                                                                     
  │           Channels 1-32: Bit predictions                   │                                                                                                      
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 5. INFERENCE                                                │                                                                                                     
  │    └─ msg_predict_inference() → Tensor[32 bits]            │                                                                                                      
  │       • Use mask to weight bit predictions                 │                                                                                                      
  │       • Spatial averaging, threshold at 0.5                │                                                                                                      
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ 6. DECODE                                                   │                                                                                                     
  │    └─ roco_decode_from_binary_tensor()                      │                                                                                                     
  │       • Extract data bits (16) + parity bits (16)          │                                                                                                      
  │       • Reed-Solomon error correction                      │                                                                                                      
  │       • Decode to ASCII (max 3 chars)                      │                                                                                                      
  │    Output: (message, is_valid, bitflips)                   │                                                                                                      
  └─────────────────────────────────────────────────────────────┘                                                                                                     
          │                                                                                                                                                           
          ▼                                                                                                                                                           
      Output: {readable_message, ecc_valid, bit_error_rate, ...}                                                                                                      
                                                                                                                                                                      
  ---                                                                                                                                                                 
  5. NEURAL NETWORK ARCHITECTURES                                                                                                                                     
                                                                                                                                                                      
  5.1 EMBEDDER (VAE-based)                                        
                                                                                                                                                                      
  VAE Encoder (Image → Latent)                                                                                                                                        
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ Input: [B, 3, 256, 256]                                    │                                                                                                      
  │                                                            │                                                                                                      
  │ Conv In: 3 → 64 channels                                   │                                                                                                      
  │                                                            │                                                                                                      
  │ Downsample Block 1:                                        │                                                                                                      
  │   └─ ResNet×2 → 64 ch                                      │                                                                                                      
  │   └─ Downsample → 128×128                                  │                                                                                                      
  │                                                            │                                                                                                      
  │ Downsample Block 2:                                        │                                                                                                      
  │   └─ ResNet×2 → 128 ch                                     │                                                                                                      
  │   └─ Downsample → 64×64                                    │                                                                                                      
  │                                                            │                                                                                                      
  │ Downsample Block 3:                                        │                                                                                                      
  │   └─ ResNet×2 → 256 ch                                     │                                                                                                      
  │   └─ No downsample → 64×64                                │                                                                                                       
  │                                                            │                                                                                                      
  │ Middle:                                                    │                                                                                                      
  │   └─ ResNet → Attention → ResNet → 256 ch                 │                                                                                                       
  │                                                            │                                                                                                      
  │ Output: [B, 4, 64, 64] (latent)                           │                                                                                                       
  └─────────────────────────────────────────────────────────────┘                                                                                                     
                │                                                                                                                                                     
                ▼                                                                                                                                                     
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ MsgProcessor (Message Injection)                            │                                                                                                     
  │                                                            │                                                                                                      
  │ Input: Latent[B,4,64,64] + Message[B,32 bits]              │                                                                                                      
  │                                                            │                                                                                                      
  │ For each bit:                                              │                                                                                                      
  │   └─ Embedding lookup → [hidden_size]                      │                                                                                                      
  │                                                            │                                                                                                      
  │ Sum all bit embeddings → [hidden_size]                     │                                                                                                      
  │ Expand to [hidden_size, 64, 64]                            │                                                                                                      
  │                                                            │                                                                                                      
  │ Concat: [B, 4+hidden_size, 64, 64]                        │                                                                                                       
  │                                                            │                                                                                                      
  │ (hidden_size = nbits × 2 = 64)                             │                                                                                                      
  └─────────────────────────────────────────────────────────────┘                                                                                                     
                │                                                                                                                                                     
                ▼                                                                                                                                                     
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ VAE Decoder (Latent → Delta Image)                          │                                                                                                     
  │                                                            │                                                                                                      
  │ Input: [B, 68, 64, 64] (4 latent + 64 msg)                 │                                                                                                      
  │                                                            │                                                                                                      
  │ Conv In: 68 → 256 channels                                 │                                                                                                      
  │                                                            │                                                                                                      
  │ Middle:                                                    │                                                                                                      
  │   └─ ResNet → Attention → ResNet → 256 ch                 │                                                                                                       
  │                                                            │                                                                                                      
  │ Upsample Block 3:                                          │                                                                                                      
  │   └─ ResNet×3 → 256 ch                                     │                                                                                                      
  │   └─ Upsample → 128×128                                   │                                                                                                       
  │                                                            │                                                                                                      
  │ Upsample Block 2:                                          │                                                                                                      
  │   └─ ResNet×3 → 128 ch                                     │                                                                                                      
  │   └─ Upsample → 256×256                                   │                                                                                                       
  │                                                            │                                                                                                      
  │ Upsample Block 1:                                          │                                                                                                      
  │   └─ ResNet×3 → 64 ch                                      │                                                                                                      
  │   └─ No upsample → 256×256                                │                                                                                                       
  │                                                            │                                                                                                      
  │ Output: [B, 3, 256, 256] (delta image)                    │                                                                                                       
  └─────────────────────────────────────────────────────────────┘                                                                                                     
                                                                                                                                                                      
  5.2 EXTRACTOR (SAM-based)                                                                                                                                           
                                                                                                                                                                      
  ImageEncoderViT (Vision Transformer - Tiny)                                                                                                                         
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ Input: [B, 3, 256, 256]                                    │                                                                                                      
  │                                                            │                                                                                                      
  │ Patch Embed: 16×16 patches → 16×16 grid                   │                                                                                                       
  │   Each patch: 3×16×16 = 768 → Project to 192 dim          │                                                                                                       
  │   Output: [B, 16, 16, 192]                                │                                                                                                       
  │                                                            │                                                                                                      
  │ Position Embedding (learned)                               │                                                                                                      
  │   Add: [B, 16, 16, 192]                                   │                                                                                                       
  │                                                            │                                                                                                      
  │ Transformer Blocks (×12 layers):                           │                                                                                                      
  │   Layer N:                                                │                                                                                                       
  │     └─ LayerNorm                                          │                                                                                                       
  │     └─ Multi-Head Attention (3 heads, windowed)           │                                                                                                       
  │     └─ Residual Connection                                │                                                                                                       
  │     └─ LayerNorm                                          │                                                                                                       
  │     └─ MLP (192 → 768 → 192, GELU)                        │                                                                                                       
  │     └─ Residual Connection                                │                                                                                                       
  │                                                            │                                                                                                      
  │ Global Attention at layers: 2, 5, 8, 11                   │                                                                                                       
  │ Window Attention elsewhere (window_size=14)                │                                                                                                      
  │                                                            │                                                                                                      
  │ Output: [B, 16, 16, 192]                                  │                                                                                                       
  └─────────────────────────────────────────────────────────────┘                                                                                                     
                │                                                                                                                                                     
                ▼                                                                                                                                                     
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ Neck (Feature Projection)                                   │                                                                                                     
  │                                                            │                                                                                                      
  │ Permute: [B, 16, 16, 192] → [B, 192, 16, 16]             │                                                                                                        
  │ Conv 1×1: 192 → 512 ch                                     │                                                                                                      
  │ LayerNorm                                                  │                                                                                                      
  │ Conv 3×3: 512 → 512 ch                                     │                                                                                                      
  │ LayerNorm                                                  │                                                                                                      
  │                                                            │                                                                                                      
  │ Output: [B, 512, 16, 16]                                  │                                                                                                       
  └─────────────────────────────────────────────────────────────┘                                                                                                     
                │                                                                                                                                                     
                ▼                                                                                                                                                     
  ┌─────────────────────────────────────────────────────────────┐                                                                                                     
  │ PixelDecoder (Feature → Predictions)                        │                                                                                                     
  │                                                            │                                                                                                      
  │ Input: [B, 512, 16, 16]                                   │                                                                                                       
  │                                                            │                                                                                                      
  │ Upsample 1:                                                │                                                                                                      
  │   └─ ConvTranspose: 512→256, scale=4 → [B,256,64,64]      │                                                                                                       
  │   └─ Conv 3×3: 256→256                                     │                                                                                                      
  │   └─ LayerNorm + ReLU                                      │                                                                                                      
  │                                                            │                                                                                                      
  │ Upsample 2:                                                │                                                                                                      
  │   └─ ConvTranspose: 256→128, scale=4 → [B,128,256,256]    │                                                                                                       
  │   └─ Conv 3×3: 128→128                                    │                                                                                                       
  │   └─ LayerNorm + ReLU                                      │                                                                                                      
  │                                                            │                                                                                                      
  │ Output Conv:                                               │                                                                                                      
  │   └─ Conv 3×3: 128 → 33 channels                           │                                                                                                      
  │                                                            │                                                                                                      
  │ Final Output: [B, 33, 256, 256]                           │                                                                                                       
  │   Channel  0: Mask prediction (sigmoid)                    │                                                                                                      
  │   Channels 1-32: Bit predictions (raw logits)              │                                                                                                      
  └─────────────────────────────────────────────────────────────┘                                                                                                     
                                                                                                                                                                      
  ---                                                                                                                                                                 
  6. KEY CLASSES AND THEIR RESPONSIBILITIES                                                                                                                           
                                                                                                                                                                      
  ┌───────────────────────┬──────────────────────────┬────────────────────────────────────────────┐
  │         Class         │           File           │               Responsibility               │                                                                   
  ├───────────────────────┼──────────────────────────┼────────────────────────────────────────────┤                                                                   
  │ WatermarkManager      │ core.py                  │ Main orchestration, embed/verify pipelines │                                                                   
  ├───────────────────────┼──────────────────────────┼────────────────────────────────────────────┤                                                                   
  │ Wam                   │ models/wam.py            │ Combines embedder, extractor, augmenter    │                                                                   
  ├───────────────────────┼──────────────────────────┼────────────────────────────────────────────┤                                                                   
  │ VAEEmbedder           │ models/embedder.py       │ Embeds message into image via VAE          │                                                                   
  ├───────────────────────┼──────────────────────────┼────────────────────────────────────────────┤                                                                   
  │ SegmentationExtractor │ models/extractor.py      │ Detects watermark via SAM                  │
  ├───────────────────────┼──────────────────────────┼────────────────────────────────────────────┤                                                                   
  │ MsgProcessor          │ modules/msg_processor.py │ Injects message into latent space          │
  ├───────────────────────┼──────────────────────────┼────────────────────────────────────────────┤                                                                   
  │ ImageEncoderViT       │ modules/vit.py           │ ViT encoder (SAM backbone)                 │
  ├───────────────────────┼──────────────────────────┼────────────────────────────────────────────┤                                                                   
  │ PixelDecoder          │ modules/pixel_decoder.py │ Decodes features to predictions            │
  ├───────────────────────┼──────────────────────────┼────────────────────────────────────────────┤                                                                   
  │ ViewframeDetector     │ viewframe_detector.py    │ Detects corner brackets                    │
  ├───────────────────────┼──────────────────────────┼────────────────────────────────────────────┤                                                                   
  │ ROCO Encoder/Decoder  │ roco_core.py             │ Message ↔ bits conversion                  │
  └───────────────────────┴──────────────────────────┴────────────────────────────────────────────┘                                                                   
                                                                  
  ---                                                                                                                                                                 
  7. CONFIGURATION PARAMETERS                                     
                                                                                                                                                                      
  Model Configuration (params.json):                              
  ├── nbits: 32                    # Message bits                                                                                                                     
  ├── img_size: 256                # Input size                                                                                                                       
  ├── scaling_w: 2.0               # Watermark strength                                                                                                               
  ├── embedder_model: vae_small    # Embedder type                                                                                                                    
  └── extractor_model: sam_base    # Extractor type                                                                                                                   
                                                                                                                                                                      
  Architecture Sizes:                                                                                                                                                 
  ├── Embedder: ~10M parameters                                                                                                                                       
  ├── Extractor: ~30M parameters (ViT tiny)                                                                                                                           
  └── Total: ~40M parameters                                                                                                                                          
                                                                                                                                                                      
  ---                                                                                                                                                                 
  8. INTERACTION DIAGRAM                                                                                                                                              
                                                                                                                                                                      
  User Application Layer                                          
          │                                                                                                                                                           
          ├── python -m mark --cover img.jpg --message ABC                                                                                                            
          │                                                                                                                                                           
          ▼                                                                                                                                                           
  WatermarkManager.embed()                                                                                                                                            
          │                                                                                                                                                           
          ├───► ROCO.encode() ───► Binary tensor [32]                                                                                                                 
          │                                                                                                                                                           
          ├───► WAM.embed()                                                                                                                                           
          │       │                                                                                                                                                   
          │       ├───► VAEEncoder                                                                                                                                    
          │       ├───► MsgProcessor                                                                                                                                  
          │       └───► VAEDecoder                                                                                                                                    
          │                                                                                                                                                           
          └───► draw_corner_brackets()                                                                                                                                
                  │                                                                                                                                                   
                  └───► Viewframe overlay


$ tree -L 3
.
├── abnormal
│   ├── s1.png
│   ├── s2.jpg
│   ├── s3.jpeg
│   └── s4.png
├── app.py
├── ARCHITECTURE.md
├── assets
│   ├── images
│   │   ├── alpaca.jpg
│   │   ├── ducks.jpg
│   │   ├── gauguin_256.jpg
│   │   ├── seabackground.jpg
│   │   └── trex_bike.jpg
│   ├── masks
│   │   ├── ducks_1.jpg
│   │   └── ducks_2.jpg
│   └── splash_wam.jpg
├── attacks
│   ├── geometric.py
│   ├── __init__.py
│   └── valuemetric.py
├── checkpoints
│   ├── params.json
│   └── wam_mit.pth
├── COMPREHENSIVE_PROJECT_REPORT.md
├── configs
│   ├── all_augs_multi_wm.yaml
│   ├── all_augs.yaml
│   ├── attenuation.yaml
│   ├── embedder.yaml
│   └── extractor.yaml
├── core.py
├── LICENSE
├── LICENSE-COCO
├── mark.py
├── notebooks
│   ├── colab.ipynb
│   ├── inference.ipynb
│   ├── inference_utils.py
│   └── __pycache__
│       ├── inference_utils.cpython-310.pyc
│       └── inference_utils.cpython-313.pyc
├── optimizer
│   ├── __init__.py
│   ├── test_log.json
│   └── wam_optimizer.py
├── __pycache__
│   ├── core.cpython-310.pyc
│   ├── core.cpython-313.pyc
│   ├── roco_core.cpython-310.pyc
│   ├── roco_core.cpython-313.pyc
│   ├── roco_ecc.cpython-310.pyc
│   ├── roco_ecc.cpython-313.pyc
│   ├── viewframe_config.cpython-310.pyc
│   ├── viewframe_config.cpython-313.pyc
│   ├── viewframe.cpython-310.pyc
│   ├── viewframe.cpython-313.pyc
│   ├── viewframe_detector.cpython-310.pyc
│   ├── viewframe_detector.cpython-313.pyc
│   ├── watermark_utils.cpython-310.pyc
│   └── watermark_utils.cpython-313.pyc
├── README.md
├── requirements.txt
├── roco_core.py
├── roco_ecc.py
├── structure.md
├── technical_report.md
├── templates
│   └── index.html
├── test
│   ├── __init__.py
│   ├── run_robustness_tests.py
│   ├── test_comprehensive_robustness.py
│   └── test_watermarking_api.py
├── test_backward_compat.py
├── test_output
│   ├── s1_final.png
│   ├── s4_diagonal_test.png
│   ├── s4_final.png
├── test_output_robustness
│   ├── full_20260330_090714
├── test_small_images.py
├── test_viewframe_improvements.py
├── verify_cli.py
├── verify.py
├── viewframe_config.py
├── viewframe_detector.py
├── viewframe.py
├── watermark_anything
│   ├── augmentation
│   │   ├── augmenter.py
│   │   ├── geometric.py
│   │   ├── __init__.py
│   │   ├── masks.py
│   │   ├── __pycache__
│   │   └── valuemetric.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── metrics.py
│   │   ├── __pycache__
│   │   └── transforms.py
│   ├── losses
│   │   ├── detperceptual.py
│   │   ├── __init__.py
│   │   ├── perceptual.py
│   │   ├── ssim.py
│   │   └── yuvloss.py
│   ├── models
│   │   ├── embedder.py
│   │   ├── extractor.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── wam.py
│   ├── modules
│   │   ├── common.py
│   │   ├── discriminator.py
│   │   ├── __init__.py
│   │   ├── jnd.py
│   │   ├── msg_processor.py
│   │   ├── pixel_decoder.py
│   │   ├── __pycache__
│   │   ├── vae.py
│   │   └── vit.py
│   └── utils
│       ├── dist.py
│       ├── image.py
│       ├── __init__.py
│       ├── logger.py
│       ├── optim.py
│       └── __pycache__
└── watermark_utils.py

44 directories, 395 files

