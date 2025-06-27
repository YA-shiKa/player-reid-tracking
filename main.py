from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import os

def main():
    print("🔄 [1] Reading input video...")
    video_path = 'input_videos/match_clip.mp4'
    if not os.path.exists(video_path):
        print(f"❌ Video not found at path: {video_path}")
        return

    video_frames = read_video(video_path)
    print(f"✅ Loaded {len(video_frames)} frames from video.")

    if not video_frames:
        print("❌ No frames found in the video.")
        return

    print("\n📦 [2] Initializing Tracker...")
    tracker = Tracker('models/yolov11_players.pt')


    print("📍 Getting object tracks (from stub if available)...")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )
    print("✅ Object tracks fetched.")
    tracker.save_id_consistency_plot(tracks)


    print("\n📌 [3] Adding positions (foot/center) to tracks...")
    tracker.add_position_to_tracks(tracks)
    print("✅ Position info added to all tracked objects.")

    print("\n🎥 [4] Estimating camera movement...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    print("✅ Camera movement estimated.")

    print("🧭 Adjusting player/ball positions based on camera movement...")
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    print("✅ Positions adjusted.")

    print("\n🌐 [5] Applying view transformation...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    print("✅ View transformed.")

    print("\n⚽ [6] Interpolating ball positions across frames...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    print("✅ Ball trajectory interpolated.")

    print("\n📊 [7] Estimating speed and distance for players...")
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    print("✅ Speed and distance values added.")

    print("\n🟥 [8] Assigning team colors...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    print("✅ Initial team colors assigned.")

    print("🎨 Getting team for each player per frame...")
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    print("✅ Team labels added to players.")

    print("\n🕹️ [9] Assigning ball possession frame-by-frame...")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_data = tracks['ball'][frame_num].get(1)
        if not ball_data:
            print(f"⚠️ Frame {frame_num}: Ball not detected.")
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)
            continue

        ball_bbox = ball_data['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            print(f"⚠️ Frame {frame_num}: Ball unassigned to any player.")
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    team_ball_control = np.array(team_ball_control)
    print("✅ Ball possession data generated.")

    print("\n🎯 [10] Drawing annotations (tracks, possession, etc.)...")
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    print("📹 Drawing camera movement visuals...")
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    print("🏃 Drawing speed and distance metrics...")
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    if not output_video_frames:
        print("❌ No output frames generated. Video will not be saved.")
        return

    print("\n💾 [11] Saving output video to: output_videos/output_video.avi")
    os.makedirs('output_videos', exist_ok=True)
    save_video(output_video_frames, 'output_videos/output_video.avi')
    print("✅ Output video saved successfully!")

if __name__ == '__main__':
    main()