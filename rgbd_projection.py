"""NumPy depth-to-color projection, independent of any reader or file format."""

import numpy as np


def _brown(x, y, coefficients, modified):
    k1, k2, p1, p2, k3 = coefficients
    r2 = x*x + y*y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2*k3))
    tx, ty = (x*radial, y*radial) if modified else (x, y)
    dx = 2*p1*tx*ty + p2*(r2 + 2*tx*tx)
    dy = 2*p2*tx*ty + p1*(r2 + 2*ty*ty)
    return radial, dx, dy


def _distortion(xy, coefficients, model, *, undistort=False):
    """Brown-Conrady variants using the RealSense SDK projection conventions."""
    coefficients = np.asarray(coefficients, dtype=np.float32)
    if not np.isfinite(coefficients).all():
        raise ValueError("Distortion coefficients must be finite")
    if not np.any(coefficients):
        return xy
    model = model.lower().replace("_", " ")
    if model not in ("brown conrady", "plumb bob", "inverse brown conrady", "modified brown conrady"):
        raise ValueError(f"Unsupported nonzero distortion model: {model}")
    if coefficients.shape != (5,):
        raise ValueError("Brown-Conrady requires five coefficients: k1,k2,p1,p2,k3")
    modified = model in ("inverse brown conrady", "modified brown conrady")
    if not undistort:
        radial, dx, dy = _brown(*xy.T, coefficients, modified)
        return xy*radial[:, None] + np.column_stack((dx,dy))
    estimate = xy.copy()
    for _ in range(10):
        radial, dx, dy = _brown(*estimate.T, coefficients, modified)
        estimate = (xy - np.column_stack((dx,dy))) / radial[:, None]
    radial, dx, dy = _brown(*estimate.T, coefficients, modified)
    estimate[np.max(np.abs(estimate*radial[:, None] + np.column_stack((dx,dy)) - xy), axis=1) > 1e-5] = np.nan
    return estimate


def build_map(depth, depth_K, color_K, depth_to_color, color_shape, *,
              depth_D=(), color_D=(), depth_model="none", color_model="none"):
    """Return float32 (H,W,2) RGB pixel coordinates indexed by depth[y,x].

    depth contains camera-axis Z in meters. K matrices are 3x3 intrinsics for
    the supplied images; no resizing or rotation is inferred. depth_to_color
    is a 4x4 transform with translation in meters, acting on column vectors.
    color_shape is (height,width). D coefficients and model names describe the
    supplied images. Unsupported nonzero distortion models raise ValueError.

    Invalid depth, points behind RGB, and out-of-image projections become NaN.
    This is a forward map only; it does not resolve color-camera occlusions.
    """
    projector = Projector(depth_K, color_K, depth_to_color, np.shape(depth), color_shape,
                          depth_D=depth_D, color_D=color_D,
                          depth_model=depth_model, color_model=color_model)
    return projector.build_map(depth)


class Projector:
    """Prepare fixed camera geometry once, then call build_map(depth) per frame.

    Inputs and output follow build_map() above. Recreate this object if image
    dimensions, calibration, or orientation change. Each result owns its array.
    """

    def __init__(self, depth_K, color_K, depth_to_color, depth_shape, color_shape, *,
                 depth_D=(), color_D=(), depth_model="none", color_model="none"):
        dk, ck, transform = [np.array(a, dtype=np.float32, copy=True)
                             for a in (depth_K,color_K,depth_to_color)]
        if dk.shape != (3,3) or ck.shape != (3,3) or transform.shape != (4,4):
            raise ValueError("Require 3x3 camera matrices and a 4x4 transform")
        if not all(np.isfinite(a).all() for a in (dk,ck,transform)):
            raise ValueError("Calibration must be finite")
        self.depth_shape, self.color_shape = tuple(depth_shape), tuple(color_shape)
        if any(len(s) != 2 or any(not isinstance(n,(int,np.integer)) or n <= 0 for n in s)
               for s in (self.depth_shape,self.color_shape)):
            raise ValueError("Image shapes must contain two positive integers")
        self.K, self.translation = ck, transform[:3,3].copy()
        self.D = np.array(color_D,dtype=np.float32,copy=True)
        _distortion(np.zeros((1,2),np.float32),self.D,color_model)  # Validate once.
        self.modified = color_model.lower().replace("_"," ") in (
            "inverse brown conrady","modified brown conrady")
        y,x = np.indices(self.depth_shape,dtype=np.float32)
        pixels = np.column_stack((x.ravel(),y.ravel(),np.ones(x.size,np.float32)))
        rays = pixels @ np.linalg.inv(dk).T
        xy = _distortion(rays[:,:2]/rays[:,2:3],depth_D,depth_model,undistort=True)
        rays = np.column_stack((xy,np.ones(x.size,np.float32))) @ transform[:3,:3].T
        self.rays = tuple(rays[:,i].reshape(self.depth_shape).copy() for i in range(3))

    def build_map(self, depth):
        depth = np.asarray(depth,dtype=np.float32)
        if depth.shape != self.depth_shape:
            raise ValueError(f"Expected depth shape {self.depth_shape}, got {depth.shape}")
        rx,ry,rz = self.rays
        tx,ty,tz = self.translation
        K = self.K
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            z = rz*depth + tz
            x,y = (rx*depth+tx)/z, (ry*depth+ty)/z
            if np.any(self.D):
                radial,dx,dy = _brown(x,y,self.D,self.modified)
                x,y = x*radial+dx, y*radial+dy
            u,v = K[0,0]*x+K[0,2], K[1,1]*y+K[1,2]
            if K[0,1] != 0:
                u += K[0,1]*y
            if K[1,0] != 0:
                v += K[1,0]*x
            if not np.array_equal(K[2],[0,0,1]):
                denominator = K[2,0]*x+K[2,1]*y+K[2,2]
                u,v = u/denominator, v/denominator
        h,w = self.color_shape
        valid = (np.isfinite(depth)&(depth>0)&(z>0)&np.isfinite(u)&np.isfinite(v)
                 &(u>=0)&(u<=w-1)&(v>=0)&(v<=h-1))
        result = np.stack((u,v),axis=2)
        result[~valid] = np.nan
        return result

    def build_maps(self, depth):
        """Return forward RGB coordinates and inverse depth pixel coordinates.

        The inverse is int32 (RGB height,width,2), with (-1,-1) for holes.
        Forward coordinates are rounded to RGB pixels; collisions retain the
        nearest color-camera Z, with equal-depth ties choosing the lowest index.
        """
        forward = self.build_map(depth)
        source = np.flatnonzero(np.isfinite(forward[:,:,0]))
        uv = np.rint(forward.reshape(-1,2)[source]).astype(np.int32)
        h,w = self.color_shape
        destination = uv[:,1]*w + uv[:,0]
        z = (self.rays[2]*np.asarray(depth,dtype=np.float32) + self.translation[2]).ravel()[source]
        # Positive float32 Z bits sort numerically. Pack Z above the source
        # index so one minimum selects the nearest point and breaks ties.
        packed = (z.view(np.uint32).astype(np.uint64) << 32) | source.astype(np.uint64)
        missing = np.iinfo(np.uint64).max
        winners = np.full(h*w,missing,dtype=np.uint64)
        np.minimum.at(winners,destination,packed)
        valid = winners != missing
        owners = (winners[valid] & np.uint64(0xffffffff)).astype(np.int32)
        inverse = np.full((h*w,2),-1,dtype=np.int32)
        inverse[valid,0] = owners % self.depth_shape[1]
        inverse[valid,1] = owners // self.depth_shape[1]
        return forward, inverse.reshape(h,w,2)


if __name__ == "__main__":
    import argparse
    from time import perf_counter
    import cv2
    from rgbd_db3_reader import RealSenseReader

    parser = argparse.ArgumentParser(description="Show a corresponding colored grid in depth and RGB.")
    parser.add_argument("bag")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--rotations", type=int, choices=range(4), default=0)
    parser.add_argument("--grid-step", type=int, default=40, help="grid spacing in depth pixels")
    parser.add_argument("--max-dimension", type=int, default=800)
    args = parser.parse_args()
    if args.grid_step <= 0 or args.max_dimension <= 0:
        parser.error("grid-step and max-dimension must be positive")
    with RealSenseReader(args.bag, rotations=args.rotations) as source:
        frame, calibration = source[args.frame], source.calibration
    started = perf_counter()
    projector = Projector(calibration.depth.K, calibration.color.K,
                        calibration.depth_to_color, frame.depth.shape, frame.color.shape[:2],
                        depth_D=calibration.depth.D, color_D=calibration.color.D,
                        depth_model=calibration.depth.distortion_model,
                        color_model=calibration.color.distortion_model)
    print(f"Projector preparation: {(perf_counter()-started)*1000:.2f} ms")
    started = perf_counter()
    mapping = projector.build_map(frame.depth)
    print(f"Map generation: {(perf_counter()-started)*1000:.2f} ms; "
          f"{np.isfinite(mapping).all(axis=2).sum():,} mapped depth pixels")

    rgb = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
    depth = cv2.applyColorMap(np.clip(frame.depth/2*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth[frame.depth <= 0] = 0
    # The same colored, numbered grid nodes mark measured correspondences.
    # Lines are skipped across invalid nodes or depth jumps greater than 5 cm.
    ys, xs = np.arange(args.grid_step//2, depth.shape[0], args.grid_step), np.arange(args.grid_step//2, depth.shape[1], args.grid_step)
    palette = [(0,255,255), (255,0,255), (255,255,0), (0,255,0)]
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            uv = mapping[y,x]
            if not np.isfinite(uv).all():
                continue
            color, label = palette[row % len(palette)], f"{row}:{col}"
            point = tuple(np.rint(uv).astype(int))
            for image, p in ((depth,(int(x),int(y))), (rgb,point)):
                cv2.circle(image,p,3,color,-1)
                cv2.putText(image,label,(p[0]+4,p[1]-4),cv2.FONT_HERSHEY_SIMPLEX,0.3,color,1)
            for nx, ny in ((x+args.grid_step,y),(x,y+args.grid_step)):
                if nx >= depth.shape[1] or ny >= depth.shape[0]:
                    continue
                target = mapping[ny,nx]
                if np.isfinite(target).all() and abs(frame.depth[ny,nx]-frame.depth[y,x]) <= 0.05:
                    cv2.line(depth,(int(x),int(y)),(int(nx),int(ny)),color,1)
                    cv2.line(rgb,point,tuple(np.rint(target).astype(int)),color,1)
    try:
        for name, image in (("Depth grid",depth),("RGB grid",rgb)):
            h,w = image.shape[:2]
            scale = args.max_dimension/max(h,w)
            image = cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
            cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
            cv2.imshow(name,image)
            cv2.moveWindow(name,20 if name == "Depth grid" else 40+args.max_dimension,40)
        print("Focus either window and press q to exit.")
        while cv2.waitKey(20) & 0xFF != ord("q"):
            try:
                closed = any(cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE) < 1 for name in ("Depth grid","RGB grid"))
            except cv2.error:
                closed = True
            if closed:
                break
    finally:
        cv2.destroyAllWindows()
