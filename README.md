# ego-hand-pose

TODO:
- [x] Github repo 
- [x] Remove bbox cls loss, 3D shape loss
- [x] Output from (29,) to (21,) for each hand -  21 per bbox, one hand per instance
- [x] Load model weight that exists other than shape reconstruction branch - use pretrained_backbone = True for pertained res-net backbone
- [x] Enabling GT 2D hand kpts - 2D hand points needed as input, originally calculated from 3D by assuming a known projection
- [ ] Replace Keypoints RCNN with POTTER 
	- Hand 2D pose estimation
