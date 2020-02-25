import numpy as np

def surface_generator(sigma, H, Lx, m, n, Lr = None, seed = None):
    #sigma = stdev (RMS roughness (Rq(m)) [SI UNITS] 1e-3 is 1mm. set to 1 to be in the same range as width and height
    #H = Hurst exponent 0 <= H <= 1 (This will define the roughness of the surface. 1 is smooth.
    #Lx = length of surface in x direction (mm)
    #m = no. of px in x 
    #n = no of px in y
       #Lr = same unit as Lx, 2*Lx/n<Lr<Lx. This is the cut off filter at which frequencies are passed through. Lr of value 2*Lx/n produces at very rough surface with much high frequency content. Converely, Lr = Lx produces a much smoother surface.
    #seed = providing an integer value as a seed will ensure that the surface topography generated is done via a repeating set of random numbers. Therefore will be the same topography regardless of whether H or Lr is changed.
    
    if Lr is not None:
        if Lr > Lx:
            raise Exception("Lr cannot be greater than Lx.")
        elif Lr < 2 * Lx / m:
            raise Exception("Increase Lr. Must be higher than 2*Lx / m.")
        else:
            qr = 2 * np.pi / Lr
    else:
        qr = 0.0
    
    if (n % 2 != 0):
        n = n - 1
    
    if (m % 2 != 0):
        m = m - 1
        
    pixelwidth = Lx / m
    
    Lx = m * pixelwidth
    Ly = n * pixelwidth
    
    qx = np.zeros((m, 1), dtype = float)
    for k in range(m):
        qx[k] = (2 * np.pi/m) * k
    
    qx = np.fft.fftshift(np.unwrap(np.fft.fftshift(qx) - np.pi)) / pixelwidth
 
    qy = np.zeros((n, 1), dtype=float)
    for j in range(n):
        qy[j] = (2 * np.pi/n) * j
        
    qy = np.fft.fftshift(np.unwrap(np.fft.fftshift(qy) - np.pi)) / pixelwidth
    
    qxx, qyy = np.meshgrid(qx, qy)

    rho = np.sqrt(qxx ** 2 + qyy ** 2)
    
    if qr is not None:
        if qr > np.pi / pixelwidth:
            raise Exception("qr greater the Nyquist frequency! Scale down qr.")
        elif qr < (2 * np.pi) / Lx:
            raise Exception("qr cannot be lower than image size. Increase qr value!")
    elif qr is None:
        qr = 0.0
    else: qr = float(qr)

    Cq = np.zeros((n, m), dtype = float)
    
    for i in range(0, m):
        for j in range(0, n):
            if (rho[j, i] < qr):
                Cq[j, i] = np.power(qr, (-2 * (H + 1)))
            else:
                Cq[j, i] = rho[j, i] ** (-2 * (H + 1))

    Cq[int(n / 2), int(m / 2)] = 0
    
    rms_f2d = np.sqrt(np.sum(Cq.sum()) * (((2 * np.pi) ** 2) / (Lx * Ly)))
                    
    alpha = sigma / rms_f2d
    
    Cq = Cq * (np.power(alpha, 2))
    
    rhof = np.floor(rho)
    
    J = 1024
    
    qrmin = np.log10(np.sqrt(np.power((2 * np.pi / Lx), 2) + np.power((2 * np.pi / Ly), 2)))
    qrmax = np.log10(np.sqrt(qx[-1] ** 2 + qy[-1] ** 2)).item()
    q = np.floor(10 ** np.linspace(qrmin, qrmax, J))

    C_AVE = np.zeros([len(q)])
    ind = np.zeros([len(q)])

    for j in range(len(q)):
        ind = np.where(np.logical_and(rhof > q[j - 1], rhof <= q[j]))
        C_AVE[j] = np.nanmean(Cq[ind])
  
    ind = ~np.isnan(C_AVE)
    C = C_AVE[ind]

    q = q[ind]
    
    Bq = np.sqrt(Cq / (np.power(pixelwidth, 2) /((n * m) * (np.power(2 * np.pi, 2)))))
    
    Bq[0, 0] = 0
    Bq[0, int(m / 2)] = 0
    Bq[int(n / 2), 0] = 0
    Bq[int(n / 2), int(m / 2)] = 0
    
    Bq[1:, 1:int(m / 2)] = Bq[1:, int(m / 2) + 1:][::-1,::-1]
    Bq[0, 1:int(m / 2)] = Bq[0, int(m / 2) + 1:][::-1]
    Bq[int(n/2) + 1:, 0] = Bq[1: int(n / 2), 0][::-1]
    Bq[int(n / 2) + 1:, int(m / 2)] = Bq[1:int(n / 2), int(m / 2)][::-1]
          
    np.random.seed(seed)        
    phi = -np.pi + (np.pi * 2) * np.random.randn(n, m).T

    phi[0, 0] = 0
    phi[0, int(m / 2)] = 0
    phi[int(n / 2), 0] = 0
    phi[int(n / 2), int(m / 2)] = 0
    
    phi[1:, 1:int(m / 2)] = -phi[1:, int(m / 2) + 1:][::-1,::-1]
    phi[0, 1:int(m / 2)] = -phi[0, int(m / 2 + 1):][::-1]
    phi[int(n / 2 + 1):, 0] = -phi[1:int(n / 2), 1][::-1]
    phi[int(n / 2 + 1):, int(m / 2)] = -phi[1:int(n / 2), int(m / 2)][::-1]
                                                   
    a = Bq * np.cos(phi)
    b = Bq * np.sin(phi)
    
    Hm = []
    for val_a, val_b in zip(a.flatten(), b.flatten()):
        Hm.append(complex(val_a, val_b))    
    Hm = np.asarray(Hm).reshape(a.shape[0], b.shape[0])
    
    surface = np.real(np.fft.ifft2(np.fft.ifftshift(Hm)))
    
    PSD = [qx, qy, Cq, q, C]
    
    return surface, pixelwidth, PSD