import numpy as np
import matplotlib.pyplot as plt
import cv2

class SkeletonVisualizer:
    def __init__(self):
        # Define the keypoint names and IDs
        self.keypoint3D_names = {
            0: 'Pelvis',
            1: 'L_Hip',
            2: 'R_Hip',
            3: 'Spine1',
            4: 'L_Knee',
            5: 'R_Knee',
            6: 'Spine2',
            7: 'L_Ankle',
            8: 'R_Ankle',
            9: 'Spine3',
            10: 'L_Foot',
            11: 'R_Foot',
            12: 'Neck',
            13: 'L_Collar',
            14: 'R_Collar',
            15: 'Head',
            16: 'L_Shoulder',
            17: 'R_Shoulder',
            18: 'L_Elbow',
            19: 'R_Elbow',
            20: 'L_Wrist',
            21: 'R_Wrist',
            22: 'L_Hand',
            23: 'R_Hand'
        }

        self.keypoint2D_names = {
            **{i: name for i, name in self.keypoint3D_names.items()},  # Include all 3D keypoints
            24: 'Head_Top',
            25: 'L_Hand_Tip',
            26: 'R_Hand_Tip',
            27: 'L_Foot_Tip',
            28: 'R_Foot_Tip',
        }
        
        # Define the keypoint kinematic tree (parent-child relationships)
        self.keypoint3D_kinematic_tree = np.array([
            [0, 1], [0, 2], [0, 3],
            [1, 4], [2, 5], [3, 6],
            [4, 7], [5, 8], [6, 9],
            [7, 10], [8, 11], [9, 12],
            [9, 13], [9, 14], [12, 15],
            [13, 16], [14, 17], [16, 18],
            [17, 19], [18, 20], [19, 21],
            [20, 22], [21, 23],
        ])
        
        self.keypoint2D_kinematic_tree = np.append(
            self.keypoint3D_kinematic_tree, # Include 3D kinematic tree
            [[15, 24], [22, 25], [23, 26], [10, 27], [11, 28]], axis=0
        )

        # Color definitions for different body parts
        self.colors = {
            'torso': (0, 255, 0),       # Green
            'right_arm': (255, 0, 0),   # Blue
            'left_arm': (0, 0, 255),    # Red
            'right_leg': (255, 255, 0), # Cyan
            'left_leg': (255, 0, 255),  # Magenta
            'head': (0, 255, 255)       # Yellow
        }
        
        # Define body part segments
        self.keypoint3D_segments = {
            'torso': [(0, 3), (3, 6), (6, 9), (9, 12), (9, 13), (9, 14)],
            'right_arm': [(14, 17), (17, 19), (19, 21), (21, 23)],
            'left_arm': [(13, 16), (16, 18), (18, 20), (20, 22)],
            'right_leg': [(0, 2), (2, 5), (5, 8), (8, 11)],
            'left_leg': [(0, 1), (1, 4), (4, 7), (7, 10)],
            'head': [(12, 15)]
        }

        self.keypoint2D_segments = {
            'torso': self.keypoint3D_segments['torso'].copy(),
            'right_arm': self.keypoint3D_segments['right_arm'].copy() + [(23, 26)],
            'left_arm': self.keypoint3D_segments['left_arm'].copy() + [(22, 25)],
            'right_leg': self.keypoint3D_segments['right_leg'].copy() + [(11, 28)],
            'left_leg': self.keypoint3D_segments['left_leg'].copy() + [(10, 27)],
            'head': self.keypoint3D_segments['head'].copy() + [(15, 24)]
        }
    
    def draw_skeleton_on_image(self, image, keypoints, thickness=2, radius=5):
        """
        Draw the skeleton on an image.
        
        Args:
            image: Input image (numpy array)
            keypoints: Array of shape (29, 2) containing 2D joint positions
            thickness: Line thickness for bones
            radius: Circle radius for joints
            
        Returns:
            Image with skeleton drawn on it
        """
        # Make a copy of the image to avoid modifying the original
        vis_img = image.copy()
        
        # If keypoints are 3D, just use the x and y coordinates
        if keypoints.shape[1] > 2:
            keypoints = keypoints[:, :2]
        
        # Convert floating point coordinates to integers
        keypoints = keypoints.astype(int)

        # Draw bones by segments (body parts)
        for part_name, segment_list in self.keypoint2D_segments.items():
            color = (self.colors[part_name][2], self.colors[part_name][1], self.colors[part_name][0])  # Convert RGB to BGR for OpenCV
            for joint_idx in segment_list:
                pt1 = tuple(keypoints[joint_idx[0]])
                pt2 = tuple(keypoints[joint_idx[1]])
                cv2.line(vis_img, pt1, pt2, color, thickness)
        
        # Draw joints (circles at each keypoint)
        for idx, point in enumerate(keypoints):
            cv2.circle(vis_img, tuple(point), radius, (255, 255, 255), -1)
            cv2.circle(vis_img, tuple(point), radius, (0, 0, 0), 1)
            
            # Optionally, add joint ID labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis_img, str(idx), (point[0] + 5, point[1] - 5), 
                        font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        return vis_img
    
    def visualize_skeleton(self, keypoints, image=None, figsize=(10, 10), title="SMPL Skeleton Visualization", figtitle=None):
        """
        Visualize the skeleton on an image or a blank canvas.
        
        Args:
            keypoints: Array of shape (29, 2) containing joint positions
            image: Optional input image (numpy array). If None, creates a blank canvas
            figsize: Figure size for matplotlib
            title: Custom title for the visualization (default: "SMPL Skeleton Visualization")
            
        Returns:
            None (displays the visualization)
        """
        # If no image is provided, create a blank canvas
        if image is None:
            height, width = 800, 800
            image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw skeleton on the image
        vis_img = self.draw_skeleton_on_image(image, keypoints)
        
        # Display using matplotlib
        plt.figure(figsize=figsize, num=figtitle)
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_3d_skeleton(self, keypoints, title="3D Human Pose", figtitle=None):
        """
        Plot the 3D skeleton using matplotlib.
        
        Args:
            keypoints: Array of shape (24, 3) containing 3D joint positions
            ax: Optional 3D axis for plotting. If None, creates a new axis
            title: Custom title for the plot (default: "3D Human Pose")
            figtitle: Optional title for the figure window (default: None)
            
        Returns:
            ax: The axis with the plotted skeleton
        """
        # Create a new 3D axis
        fig = plt.figure(num=figtitle)
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw bones by segments (body parts)
        for part_name, segment_list in self.keypoint3D_segments.items():
            color = tuple(c / 255.0 for c in self.colors[part_name])  # Normalize color to [0, 1]
            for joint_idx in segment_list:
                pt1 = keypoints[joint_idx[0]]
                pt2 = keypoints[joint_idx[1]]
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], color=color, linewidth=2)

        # Draw joints (spheres at each keypoint)
        for point in keypoints:
            ax.scatter(point[0], point[1], point[2], color='black', s=50) # type: ignore
        
        # Add axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') # type: ignore
        ax.set_title(title)  # Use the provided title parameter

        ax.set_xlim((-2.0, 1.6))
        ax.set_ylim((-1.0, 1.2))

        # Equal aspect ratio for all axes
        ax.set_box_aspect((1, 1, 1)) # type: ignore

        ax.view_init(elev=-90, azim=-90) # type: ignore

        # plt.tight_layout()
        plt.show()

    