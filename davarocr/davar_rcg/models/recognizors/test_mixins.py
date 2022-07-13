"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    rf_learning.py
# Abstract       :    Implementations of the RF-leanring Recognizor Structure

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""

import Levenshtein 


class TextRecognitionTestMixin(object):

    def aug_test_text_recognition(self, imgs, gt_texts, **kwargs):
        """Simple test for mask head without augmentation."""
        # image shapes of images in the batch
        result = []
        for i, img in enumerate(imgs):
            result.append(self.simple_test(img, gt_texts, **kwargs))
        return result
    
    def merge_string(self, s1, s2):
        m = min(len(s1), len(s2))
        for i in range(m, 0, -1):
            # Compare whether the last i character of s1 is the same as the first i character of s2
            if s1[-i:] == s2[:i] or Levenshtein.distance(s1[-i:], s2[:i]) < 2:
                return s1 + s2[i:]
        return ""


    def post_processing(self, string_list):
        template_string = string_list[0]['text'][0]
        if len(string_list) > 1:
            for string in string_list[1:]:
                if len(string['text'][0]):
                    template_string = self.merge_string(template_string, string['text'][0])
                else:
                    continue
        return template_string
                


