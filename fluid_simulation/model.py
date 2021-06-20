from numba import cuda
import numba
import numpy as np
import cloudsProcessor
from time import time
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

Steps = 1000000
Size = 1024*2*2*2*2 #areaSize
Count = 1
IEC = 0.01

def flowMapGenerate(source_map, InitConcentration = IEC, nstp = 1000, n = Size, iCount = Count):
    
    InitMatterConcentration_Map, InitDashConcentration_Map = cloudsProcessor(source_map, InitConcentration)
    
    blockdim = 16, 16
    griddim = int(n / blockdim[0]), int(n / blockdim[1])

    DeltaT = 0.01
    Volume = 0.1 * (n * n) / (256 * 256)
    GrowthSpeed = 0.1

    DifMatter = 0.01
    DifDash = 0.01

    Width = Height = n

    GlobalCoef = 1.0
    RoMatter = 1.0
    RoDash = 1.0
    winSize = 1
    DiffuseSteps = 1.0

    v0 = GlobalCoef * Volume / (Width * Height)
    h = (0.01) / (math.pi * 1 * 1)

    DeltaX = math.sqrt(Volume / h) / Height

    MaxMatter = RoMatter * v0 
    MaxDash = RoDash * v0 
    EquMatter = InitConcentration * v0 

    SpeedMaxDash = MaxDash 

    CrystlProbSpeed = - GrowthSpeed * ((MaxMatter - EquMatter) * DeltaT) / (EquMatter * DeltaX)
    DissolveProbSpeed = - GrowthSpeed * (EquMatter * DeltaT) / ((MaxMatter - EquMatter) * DeltaX)

    DiffuseMatterSpeed = DifMatter * DeltaT / ((DeltaX * DeltaX) * DiffuseSteps * MaxDash)
    DiffuseDashSpeed = DifDash * DeltaT / ((DeltaX * DeltaX) * DiffuseSteps * MaxMatter)

    OptimalT = pow(DeltaX, 2) / (4 * max(DifMatter, DifDash))
    print("Max DT =", OptimalT)
    cuda.select_device(1)

    #CUDA Diffuse
    @cuda.jit('void(float32[:], float32[:])')
    def Diffuse_Copy(u0, u):
        i, j = cuda.grid(2)
        u0[i + j * n] = u[i + j * n]
        
    @cuda.jit('void(float32[:], float32[:], float32[:], float32[:])')
    def Diffuse_Matter_1d_stp_gpu(uB, uM, uD, d_map):
        i, j = cuda.grid(2)
        uB[i + j * n] = uM[i + j * n]
        
        if d_map[i + j * n] < 1:
            
            defval = uM[i + j * n]
            uim1 = uip1 = ujm1 = ujp1 = defval
            
            if i > 0 and d_map[i - 1 + j * n] < 1:
                uim1 = uM[i - 1 + j * n]
                
            if i < n - 1 and d_map[i + 1 + j * n] < 1:
                uip1 = uM[i + 1 + j * n]
                
            if j > 0 and d_map[i + (j - 1) * n] < 1:
                ujm1 = uM[i + (j - 1) * n]
                
            if j < n - 1 and d_map[i + (j + 1) * n] < 1:
                ujp1 = uM[i + (j + 1) * n]

            if (MaxDash > uD[i + j * n]):
                uB[i + j * n] = defval + (uim1 + uip1 + ujm1 + ujp1 - 4. * defval) * DiffuseMatterSpeed * (MaxDash - uD[i + j * n])

            
    @cuda.jit('void(float32[:], float32[:], float32[:], float32[:])')
    def Diffuse_Dash_1d_stp_gpu(uB, uD, uM, d_map):
        i, j = cuda.grid(2)
        
        uB[i + j * n] = uD[i + j * n]
        
        if d_map[i + j * n] < 1:
            
            defval = uD[i + j * n]
            uim1 = uip1 = ujm1 = ujp1 = defval
            
            if i > 0 and d_map[i - 1 + j * n] < 1:
                uim1 = uD[i - 1 + j * n]
                
            if i < n - 1 and d_map[i + 1 + j * n] < 1:
                uip1 = uD[i + 1 + j * n]
                
            if j > 0 and d_map[i + (j - 1) * n] < 1:
                ujm1 = uD[i + (j - 1) * n]
                
            if j < n - 1 and d_map[i + (j + 1) * n] < 1:
                ujp1 = uD[i + (j + 1) * n]

            if (MaxMatter > uM[i + j * n]):
                uB[i + j * n] = defval + (uim1 + uip1 + ujm1 + ujp1 - 4. * defval) * DiffuseDashSpeed * (MaxMatter - uM[i + j * n])
            

    #CUDA Probability
    @cuda.jit('void(float32[:], float32[:], float32[:], float32[:])')
    def CalcProbability1(uM, uN, uD, uF):
        i, j = cuda.grid(2)
        
        if  uF[i + j * n] > 0:
            uD[i + j * n] = 1
            uN[i + j * n] = MaxMatter - uM[i + j * n]
            uM[i + j * n] = MaxMatter     
        else:   
            uN[i + j * n] = 0

        if uD[i + j * n] > 0:
            uF[i + j * n] = 1
        else:
            uF[i + j * n] = 0
            
    @cuda.jit
    def CalcProbability0(uM, uD, uF, rng_states, result):
        i, j = cuda.grid(2)
        
        uF[i + j * n] = 0

        if i > winSize and i < n - winSize:
            if j > winSize and j < n - winSize:
                if uD[i + j * n] < 1:
                    flag = 0
                    for ii in range(i - winSize, i + winSize + 1):
                        for jj in range(j - winSize, j + winSize + 1):
                            if uD[ii + jj * n] > 0:
                                flag = 1
                                break
                        if flag > 0:
                            break
                    
                    if flag:
                    
                        CPmatter = uM[i + j * n]

                        isKeyPoint = 0
                        if (CPmatter <= 0):
                            isKeyPoint = 0
                        elif (CPmatter >= MaxMatter):
                            isKeyPoint = 1
                        else:
                            p = 1 - math.exp(CrystlProbSpeed * CPmatter / (MaxMatter - CPmatter))
                            if p > xoroshiro128p_uniform_float32(rng_states, i + j * n ):
                                isKeyPoint = 1
                        
                        isDissolve = 0
                        if (CPmatter <= 0):
                            isDissolve = 1
                        elif (CPmatter >= MaxMatter):
                            isDissolve = 0
                        else:
                            p = 1 - math.exp(DissolveProbSpeed * (MaxMatter - CPmatter) / CPmatter) 
                            if p > xoroshiro128p_uniform_float32(rng_states, i + j * n):
                                isDissolve = 1

                        if isKeyPoint > 0 and isDissolve < 1:
                            uF[i + j * n] = 1
                            if i <= 5:
                                cuda.atomic.add(result, 0, 1.)
                            if j <= 5:
                                cuda.atomic.add(result, 1, 1.)
                            if i >= n - 5:
                                cuda.atomic.add(result, 2, 1.)
                            if j >= n - 5:
                                cuda.atomic.add(result, 3, 1.)


    #CUDA MatterGoing
    @cuda.jit('void(float32[:], float32[:], float32[:])')
    def EnouthMatter(uM, uD, result):    
        i, j = cuda.grid(2)
        if(uD[i + j * n] > 0):
            cuda.atomic.add(result, 0, 1.)
        cuda.atomic.add(result, 1, uM[i + j * n])    

    #CUDA MatterGoing
    @cuda.jit('void(float32[:], float32[:], float32[:])')
    def MatterGoing0(uM, uN, uF):
        i, j = cuda.grid(2)      

        if (uN[i + j * n] > 0):
            if (1 > 0):
                num = 1
                matterNum = 0
                for winI in range(i - num, i + num + 1):
                    for winJ in range(j - num, j + num + 1):
                        if (abs(winI - i) + abs(winJ - j) == num):                                  
                            if winI > 0 and winI < n - 1 and winJ > 0 and winJ < n - 1:            
                                if uF[winI + winJ * n] < 1:                                         
                                    matterNum += 1               
        
                for winI in range(i - num, i + num + 1):
                    for winJ in range(j - num, j + num + 1):
                        if (abs(winI - i) + abs(winJ - j) == num):                                  
                            if winI > 0 and winI < n - 1 and winJ > 0 and winJ < n - 1:             
                                if uF[winI + winJ * n] < 1:                                         
                                    cuda.atomic.add(uM, winI + winJ * n, -1. * uN[i + j * n] / (1.*matterNum)) 
                uN[i + j * n] = 0


    @cuda.jit('void(float32[:], float32[:], float32[:])')
    def MatterGoing1(uM, uN, uF):
        i, j = cuda.grid(2)
        
        if (uM[i + j * n] < 0):
            uF[i + j * n] = 1
            uN[i + j * n] = -1. * uM[i + j * n]
            uM[i + j * n] = 0

    @cuda.jit('void(float32[:], float32[:])')
    def MatterGoing2(uM, uN):
        i, j = cuda.grid(2)

        uN[i + j * n] = 0

        if (uM[i + j * n] < 0):
            uM[i + j * n] = 0
        elif (uM[i + j * n] > MaxMatter):
            uM[i + j * n] = MaxMatter

    for countI in range (0,iCount):
        if (OptimalT < DeltaT):
            print('Terminated')
            break
        print('working...')
        #programStart
        st = time() #timer
        
        Matter = cv2.resize(InitMatterConcentration_Map, dsize=(n, n), interpolation=cv2.INTER_CUBIC)
        Dash = cv2.resize(InitDashConcentration_Map, dsize=(n, n), interpolation=cv2.INTER_CUBIC)
        
        Buffer = np.full((n * n), 0., dtype = np.float32)                               #DataGeneration
        PointsMap = np.full((n * n), 0., dtype = np.float32)                            #DataGeneration
        TempMap = np.full((n * n), 0., dtype = np.float32)                              #DataGeneration
        rng_states = create_xoroshiro128p_states(n * n, seed = countI)                  #RNG
        result = np.zeros(4, dtype = np.float32)
        #seed
        FirstCoord = int(n / 2)
        Matter[FirstCoord+FirstCoord * n] = MaxMatter
        Dash[FirstCoord+FirstCoord * n] = 0
        PointsMap[FirstCoord+FirstCoord * n] = 1

        #CopyToCard
        d_Matter = cuda.to_device(Matter)
        d_Dash = cuda.to_device(Dash)
        d_Buffer = cuda.to_device(Buffer)
        d_PointsMap = cuda.to_device(PointsMap)
        d_TempMap = cuda.to_device(TempMap)
        d_rng_states = cuda.to_device(rng_states)
        d_Result = cuda.to_device(result)

        result[0] = result[1] = result[2] = result[3] = 0
        #Process
        for i in range(0, nstp):
            
            CalcProbability0[griddim, blockdim](d_Matter, d_PointsMap, d_TempMap, d_rng_states, result)
            CalcProbability1[griddim, blockdim](d_Matter, d_Buffer, d_PointsMap, d_TempMap)
            if (1 > 0):      
                x = 0
                while x < 10:
                    x = x + 1
                    MatterGoing0[griddim, blockdim](d_Matter, d_Buffer, d_TempMap)
                    MatterGoing1[griddim, blockdim](d_Matter, d_Buffer, d_TempMap)

            MatterGoing2[griddim, blockdim](d_Matter, d_Buffer)
               
            #Diffuse
            Diffuse_Matter_1d_stp_gpu[griddim, blockdim](d_Buffer, d_Matter, d_Dash, d_PointsMap)
            #cuda.synchronize()        
            Diffuse_Copy[griddim, blockdim](d_Matter, d_Buffer)
            #cuda.synchronize()
            Diffuse_Dash_1d_stp_gpu[griddim, blockdim](d_Buffer, d_Dash, d_Matter, d_PointsMap)
            #cuda.synchronize()
            Diffuse_Copy[griddim, blockdim](d_Dash, d_Buffer)
            #cuda.synchronize()
            if result[0] > 0 and result[1] > 0 and result[2] > 0 and result[3] > 0:
                break
        
        #timer
        Matter = None       #DataGeneration
        Dash = None         #DataGeneration
        Buffer = None       #DataGeneration
        PointsMap = None      #DataGeneration
        TempMap = None      #DataGeneration
        rng_states = None   #DataGeneration
        d_Matter = None
        d_Dash = None
        d_Buffer = None
        d_PointsMap = None
        d_TempMap = None
        
        print ('Generation ', str(countI + 1), "/",  str(iCount),' complite on Step =', i + 1, ';   time =', time() - st)

        return PointsMap
