"""
@author: t.f.tiotto@rug.nl
@university: University of Groningen
@group: Cognitive Modeling

Function:
-

Script arguments:
-

Script output:
-
"""


def generate_video( folder, population ):
    import os
    
    folder_snapshots = folder + '/' + population + '_syn_matrix'
    
    ffmpeg_string = "ffmpeg " + "-pattern_type glob -i '" \
                    + folder_snapshots + "/" + "*.png' " \
                                               "-c:v libx264 -preset veryslow -crf 17 " \
                                               "-tune stillimage -hide_banner -loglevel warning " \
                                               "-y -pix_fmt yuv420p " \
                                               "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' " \
                    + folder + "/syn_matrix_heatmaps.mp4"
    
    print( f"Generating Video from Heatmaps using {ffmpeg_string} ..." )
    os.system( ffmpeg_string )
    print( f"Generated video {folder + '/syn_matrix_heatmaps.mp4'}" )
