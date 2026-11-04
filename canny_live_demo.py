"""Live Canny tuning: S pause/resume, G save a snapshot, Q/Esc quit."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from rgbd_live import RealSenseLive


def edges_for(rgb, low, high, blur, aperture, l2):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if blur:
        gray = cv2.GaussianBlur(gray, (2*blur+1, 2*blur+1), 0)
    return cv2.Canny(gray, min(low, high), max(low, high),
                     apertureSize=3+2*aperture, L2gradient=bool(l2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial')
    parser.add_argument('--max-dimension', type=int, default=600)
    parser.add_argument('--rotations', type=int, choices=range(4), default=1)
    parser.add_argument('--output-dir', type=Path, default=Path('canny_snapshots'))
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int, default=0)
    args = parser.parse_args()
    if args.max_dimension <= 0 or args.frames < 0:
        parser.error('max-dimension must be positive; frames must be nonnegative')
    cv2.setNumThreads(1)
    sliders = {'Low': 50, 'High': 150, 'Blur radius': 1, 'Aperture 3/5/7': 0, 'L2 gradient': 1}
    windows = ('Canny overlay', 'Canny edges')
    controls = 'Canny controls'
    if not args.headless:
        for window in windows:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(controls, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(controls, np.zeros((40, 600, 3), np.uint8))
        for name, maximum in zip(sliders, (1000, 1000, 5, 2, 1)):
            cv2.createTrackbar(name, controls, sliders[name], maximum, lambda value: None)
        for window, x in zip((*windows, controls), (0, 350, 700)):
            cv2.moveWindow(window, x, 30)
    paused, count = False, 0
    print('S pause/resume | G save RGB, edges and settings | Q/Esc quit', flush=True)
    try:
        with RealSenseLive(serial=args.serial, rotations=args.rotations) as source:
            print(f'{source.name} {source.serial}', flush=True)
            while not args.frames or count < args.frames:
                if not paused or count == 0:
                    frame = source.read()
                if not args.headless:
                    sliders = {name: cv2.getTrackbarPos(name, controls) for name in sliders}
                started = perf_counter()
                edges = edges_for(frame.color, *sliders.values())
                elapsed = (perf_counter()-started)*1000
                overlay = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
                overlay[edges != 0] = (0, 255, 0)
                count += 1
                key = -1
                if not args.headless:
                    for window, image in zip(windows, (overlay, edges)):
                        h, w = image.shape[:2]
                        scale = min(1, args.max_dimension/max(h, w))
                        view = cv2.resize(image, (round(w*scale), round(h*scale)))
                        if window == windows[0]:
                            cv2.putText(view, f'{elapsed:.1f} ms | {"PAUSED" if paused else "LIVE"} | S pause G save Q quit',
                                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                        cv2.imshow(window, view)
                    key = cv2.waitKey(15) & 255
                    if key in (27, ord('q')) or any(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in windows):
                        break
                    if key == ord('s'):
                        paused = not paused
                if key == ord('g') or (args.headless and count == args.frames):
                    folder = args.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    folder.mkdir(parents=True)
                    cv2.imwrite(str(folder/'rgb.png'), cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(folder/'edges.png'), edges)
                    cv2.imwrite(str(folder/'overlay.png'), overlay)
                    settings = dict(sliders, processing_ms=elapsed, timestamp_ns=frame.timestamp_ns,
                                    rotations=args.rotations, blur_kernel=2*sliders['Blur radius']+1,
                                    aperture=3+2*sliders['Aperture 3/5/7'])
                    (folder/'settings.json').write_text(json.dumps(settings, indent=2))
                    print(f'Saved {folder}', flush=True)
                if count % 60 == 0:
                    print(f'{count}: Canny {elapsed:.1f} ms', flush=True)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
