"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    generate_html.py
# Abstract       :    generate html from area, which is the intermediate result of post-processing.

# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

from html import escape
import numpy as np


def area_to_html(area, labels, texts_tokens):
    """ Generate structure html and text tokens from area, which is the intermediate result of post-processing.

    Args:
        area(np.array): (n x n). Two-dimensional array representing the distribution of rows and columns for each cell.
        labels(list[list]): (n x 1).labels of each non-empty cell
        texts_tokens(list[list]): texts_tokens for each non-empty cell

    Returns:
        list(str): The html that characterizes the structure of table
        list(str): text tokens for each cell (including empty cells)
    """

    area_extend = np.zeros([area.shape[0] + 1, area.shape[1] + 1])
    area_extend[:-1, :-1] = area
    html_struct_recon = []
    text_tokens_recon = []
    headend = 0
    for height in range(area.shape[0]):
        html_struct_recon.append("<tr>")
        width = 0
        numhead, numbody = 0, 0
        while width < area.shape[1]:
            # curent cell is rest part of a rowspan cell
            if height != 0 and area_extend[height, width] == area_extend[height - 1, width]:
                width += 1

            # td without span
            elif area_extend[height, width] != area_extend[height + 1, width] and area_extend[height, width] != \
                    area_extend[height, width + 1]:
                html_struct_recon.append("<td>")
                html_struct_recon.append("</td>")
                texts_insert = texts_tokens[int(area_extend[height, width]) - 1] if int(
                    area_extend[height, width]) >= 1 else [""]
                text_tokens_recon.append({'tokens': texts_insert})

                # caculate the number of "</thead>" and "<body>" in this row
                if int(area_extend[height, width]) < 1:
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += 1

            # td only with colspan
            elif area_extend[height, width] != area_extend[height + 1, width] and area_extend[height, width] == \
                    area_extend[height, width + 1]:
                colspan = 1
                while area_extend[height, width] == area_extend[height, width + colspan]:
                    colspan += 1
                    if (width + colspan) == (area.shape[1]):
                        break
                html_struct_recon.append("<td")
                html_struct_recon.append(" colspan=\"%s\"" % str(colspan))
                html_struct_recon.append(">")
                html_struct_recon.append("</td>")
                texts_insert = texts_tokens[int(area_extend[height, width]) - 1] if int(
                    area_extend[height, width]) >= 1 else [""]
                text_tokens_recon.append({'tokens': texts_insert})

                # caculate the number of "</thead>" and "<body>" in this row
                if int(area_extend[height, width]) < 1:
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += colspan

            # td only with rowspan
            elif area_extend[height, width] == area_extend[height + 1, width] and area_extend[height, width] != \
                    area_extend[height, width + 1]:
                rowspan = 1
                while area_extend[height, width] == area_extend[height + rowspan, width]:
                    rowspan += 1
                    if height + rowspan == area.shape[0]:
                        break
                html_struct_recon.append("<td")
                html_struct_recon.append(" rowspan=\"%s\"" % str(rowspan))
                html_struct_recon.append(">")
                html_struct_recon.append("</td>")
                texts_insert = texts_tokens[int(area_extend[height, width]) - 1] if int(
                    area_extend[height, width]) >= 1 else [""]
                text_tokens_recon.append({'tokens': texts_insert})

                # caculate the number of "</thead>" and "<body>" in this row
                if int(area_extend[height, width]) < 1:
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += 1

            # td with row span and col span togther
            elif area_extend[height, width] == area_extend[height + 1, width] and area_extend[height, width] == \
                    area_extend[height, width + 1]:
                rowspan = 1
                while area_extend[height, width] == area_extend[height + rowspan, width]:
                    rowspan += 1
                    if height + rowspan == area.shape[0]:
                        break
                html_struct_recon.append("<td")
                html_struct_recon.append(" rowspan=\"%s\"" % str(rowspan))
                colspan = 1
                while area_extend[height, width] == area_extend[height, width + colspan]:
                    colspan += 1
                    if (width + colspan) == (area.shape[1]):
                        break
                html_struct_recon.append(" colspan=\"%s\"" % str(colspan))
                html_struct_recon.append(">")
                html_struct_recon.append("</td>")
                texts_insert = texts_tokens[int(area_extend[height, width]) - 1] if int(
                    area_extend[height, width]) >= 1 else [""]
                text_tokens_recon.append({'tokens': texts_insert})

                # caculate the number of "</thead>" and "<body>" in this row
                if int(area_extend[height, width]) < 1:
                    pass
                elif labels[int(area_extend[height, width]) - 1][0]:
                    numbody += 1
                elif not labels[int(area_extend[height, width]) - 1][0]:
                    numhead += 1
                width += colspan

        html_struct_recon.append("</tr>")
        if numhead > numbody:
            headend = height + 1

    # insert '<thead>', '</thead>', '<tbody>' and '</tbody>'
    rowindex = [ind for ind, td in enumerate(html_struct_recon) if td == '</tr>']
    if headend:
        html_struct_recon.insert(rowindex[headend - 1] + 1, '</thead>')
        html_struct_recon.insert(rowindex[headend - 1] + 2, '<tbody>')
    else:
        trindex = html_struct_recon.index('</tr>')
        html_struct_recon.insert(trindex + 1, '</thead>')
        html_struct_recon.insert(trindex + 2, '<tbody>')
    html_struct_recon.insert(0, '<thead>')
    html_struct_recon.append('</tbody>')

    return html_struct_recon, text_tokens_recon


def format_html(html_struct, text_tokens):
    """ Formats HTML code from structure html and text tokens

    Args:
        html_struct (list(str)): structure html
        text_tokens (list(dict)): text tokens

    Returns:
        str: The final html of table.
    """

    html_code = html_struct.copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], text_tokens[::-1]):
        if cell['tokens']:
            cell = [escape(token) if len(token) == 1 else token for token in cell['tokens']]
            cell = ''.join(cell)
            html_code.insert(i + 1, cell)
    html_code = ''.join(html_code)
    html_code = '''<html><body><table>%s</table></body></html>''' % html_code

    return html_code
