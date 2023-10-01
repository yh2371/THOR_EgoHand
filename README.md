# ego-hand-pose

TODO:
0. Github repo - [x]
1. Remove bbox cls loss, 3D shape loss - [x]
2. Output from (29,) to (21,) for each hand - [x] 21 per bbox, one hand per instance
3. Load model weight that exists other than shape reconstruction branch - [x] use pretrained_backbone = True for pertained res-net backbone
4. Enabling GT 2D hand kpts - [x] 2D hand points needed as input, originally calculated from 3D by assuming a known projection
5. Replace Keypoints RCNN with POTTER - [ ]
	- Hand 2D pose estimation