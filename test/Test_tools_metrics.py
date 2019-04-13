# coding: utf-8

# coding: utf-8
import imagedt

class Test_Tools_Metrics(object):
    def __init__(self):
      super(Test_Tools_Metrics, self).__init__()

    def test_f1_score(self, predict_file, gt_file, labelt_file):
      imagedt.tools.metrics.calculate_f1_score(predict_file, gt_file, labelt_file)

    def test_roc(self, predict_image_infos, show_fig=False):
      imagedt.tools.metrics.roc(predict_image_infos, show_fig=show_fig)

    def test_generate_capt(self):
      TG = imagedt.tools.text_generator.Text_Generator(
        font_path='/data/project/idt-pitaya-serv-tensorflow/tf_utils/fonts/English_font',
        langu='en', 
        )
      # font_type: Chinese_font ,English_font
      generate_items = TG.gen_price_realtime_datas(batch_size=64)
      print (generate_items)


if __name__ == '__main__':
  Test_Metrics = Test_Tools_Metrics()
  # gt_file line:[image name, predict class] 219549_0.jpg,723
  # predict_file line: [image name, predict class, confidence] 219549_0.jpg,723,0.999
  # labelt_file line: [object_id, cls_id, cls_name] :6973 0 维他奶维他柠檬茶250ml盒装_6973
  # gt_file = '/ssd_data/danon/meeting_shelfs/test_outinfos/label_infos.txt'
  # predict_file = '/ssd_data/danon/meeting_shelfs/test_outinfos/test_infos.txt'
  # labelt_file = '/ssd_data/danon/cls_records/labels_t.txt'
  # # Test_Metrics.test_f1_score()

  # # predict_image_infos line format: Image_name,True_class_name,Label_id,Predict_id,Confidence,Match
  # predict_image_infos = '/ssd_data/danon/meeting_shelfs/test_outinfos/image_classify_infos.csv'
  # Test_Metrics.test_roc(predict_image_infos, show_fig=True)


  Test_Metrics.test_generate_capt()
  

