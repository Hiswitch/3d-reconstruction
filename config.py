confs = {
    "superpoint_aachen": {
        "output": "feats-superpoint-n4096-r1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
        },
    },
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    "superpoint_max": {
        "output": "feats-superpoint-n4096-rmax1600",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
            "resize_force": True,
        },
    },
    "superpoint_inloc": {
        "output": "feats-superpoint-n4096-r1600",
        "model": {
            "name": "superpoint",
            "nms_radius": 4,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "r2d2": {
        "output": "feats-r2d2-n5000-r1024",
        "model": {
            "name": "r2d2",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
        },
    },
    "d2net-ss": {
        "output": "feats-d2net-ss",
        "model": {
            "name": "d2net",
            "multiscale": False,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    "sift": {
        "output": "feats-sift",
        "model": {"name": "dog"},
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "sosnet": {
        "output": "feats-sosnet",
        "model": {"name": "dog", "descriptor": "sosnet"},
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "disk": {
        "output": "feats-disk",
        "model": {
            "name": "disk",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
    # Global descriptors
    "dir": {
        "output": "global-feats-dir",
        "model": {"name": "dir"},
        "preprocessing": {"resize_max": 1024},
    },
    "netvlad": {
        "output": "global-feats-netvlad",
        "model": {"name": "netvlad"},
        "preprocessing": {"resize_max": 1024},
    },
    "openibl": {
        "output": "global-feats-openibl",
        "model": {"name": "openibl"},
        "preprocessing": {"resize_max": 1024},
    },
    "eigenplaces": {
        "output": "global-feats-eigenplaces",
        "model": {"name": "eigenplaces"},
        "preprocessing": {"resize_max": 1024},
    },
}