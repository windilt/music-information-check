# music-information-check
check the music file or midi file information

20250330

  程序基本是由DeepSeek V3写的
  
  程序的主要目的是获取歌曲的基本信息，用来给音乐小白扒谱/转调提供帮助。具体信息包括
  
    BPM
    
    采样率
    
    时长
    
    各个音符出现的百分比
    
    由直方图表示的每个音符出现次数，并且跟人声的音域做了比较
    
  如果扒旋律谱的话，输入最好是midi文件，其次是去除了和声和混响的人声（可以参考MSST-WebUI项目）
