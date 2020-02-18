import sys
import shutil

filename = sys.argv[1]
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])

# Go through xml files for every frame in range
for frame_no in range(start_frame, end_frame + 1):
    print(f'Recovering annotation for frame {frame_no} of {filename}')
    output_path = f'../dataset/behaviour_annotations/{filename}'
    shutil.copy(f'../dataset/behaviour_annotations_backup/{filename}/{filename}_frame_{frame_no}.xml', output_path)
