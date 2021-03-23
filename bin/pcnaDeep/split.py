# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import skimage.measure as measure

def split_frame(frame, n=4):
    """Split frame into several quadrants

    Args:
        frame (numpy.array): single frame slice to split, shape HWC, if HW, will expand C
        n (int): split count, either 4 or 9

    Returns:
        (numpy.array): stack of splitted slice, order by row
    """
    if n not in [4,9]:
        raise ValueError('Split number should be 4 or 9.')
    
    if frame.shape[0] != frame.shape[1]:
        raise ValueError('Frame should be square.')

    if len(frame.shape)<3:
        frame = np.expand_dims(frame, axis=2)
    if frame.shape[0]/n != int(frame.shape[0]/n):
        pd_out = (frame.shape[0]//n + 1) * n - frame.shape[0]
        frame = np.pad(frame, ((0,pd_out), (0,pd_out),(0,0)), 'constant', constant_values=(0,))

    row = np.split(frame, np.sqrt(n), axis=0)
    tile = []
    for r in row:
        tile.extend(np.split(r, np.sqrt(n), axis=1))

    return np.stack(tile, axis=0)


def join_frame(stack, n=4, crop_size=None):
    """For each n frame in the stack, join into one complete frame (by row)

    Args:
        stack (numpy.array): tiles to join
        n (int): each n tiles to join, should be either 4 or 9.
        crop_size (int): crop the square image into certain size (lower-right), 
            default no crop

    Returns:
        (numpy.array): stack of joined frames
    """

    if n not in [4,9]:
        raise ValueError('Join tile number should either be 4 or 9.')

    if stack.shape[0] < n or stack.shape[0] % n != 0:
        raise ValueError('Stack length is not multiple of tile count n.')

    p = int(np.sqrt(n))
    out_stack = []
    stack = stack.astype('uint16')
    for i in range(int(stack.shape[0]/n)):
        count = 1
        frame = []
        for j in range(p):
            row = []
            for k in range(p):
                new_stack, count_add = relabel_seq(stack[int(j * p + k + i * n),:], base=count)
                count += count_add
                row.append(new_stack)
            row = np.concatenate(np.array(row), axis=1)
            frame.append(row)
        frame = np.concatenate(np.array(frame), axis=0)
        out_stack.append(frame)
    
    out_stack = np.stack(out_stack, axis=0)

    if crop_size is not None:
        out_stack = out_stack[:, :crop_size, :crop_size, :]
    
    if np.max(stack) <= 255:
        out_stack = out_stack.astype('uint8')
    else:
        out_stack = out_stack.astype('uint16')

    return out_stack


def join_table(table, n=4, tile_width=1200):
    """Join object table according to tiled frames

    Args:
        table (pandas.DataFrame): object table to join, 
            essential columns: frame, Center_of_the_object_0 (x), Center_of_the_object_1 (y).
            The method will join frames by row.
        n (int): each n frames form a tiled slice, either 4 or 9
        tile_width (int): width of each tile
        
    """
    NINE_DICT = {0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 
                5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)}
    FOUR_DICT = {0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)}
    if n not in [4,9]:
        raise ValueError('Join tile number should either be 4 or 9.')
    
    if (np.max(table['frame'])+1) < n or (np.max(table['frame']) + 1) % n != 0:
        raise ValueError('Stack length is not multiple of tile count n.')    

    out = pd.DataFrame()
    for i in range(int(np.max(table['frame'])/n)):
        sub_table = table[(table['frame']<(i+1)*n) & (table['frame']>= (i*n))].copy()
        mod_x = []
        mod_y = []
        for j in range(n):
            sub_tile_table = sub_table[sub_table['frame']==(i*n+j)].copy()
            for k in range(sub_tile_table.shape[0]):
                x = sub_tile_table['Center_of_the_object_0'].iloc[k]
                y = sub_tile_table['Center_of_the_object_1'].iloc[k]
                if n == 4:
                    x += FOUR_DICT[j][0] * tile_width
                    y += FOUR_DICT[j][1] * tile_width
                else:
                    x += NINE_DICT[j][0] * tile_width
                    y += NINE_DICT[j][1] * tile_width
                mod_x.append(x)
                mod_y.append(y)
        sub_table.loc[:,'Center_of_the_object_0'] = mod_x
        sub_table.loc[:,'Center_of_the_object_1'] = mod_y
        out = out.append(sub_table)

    return out


def relabel_seq(frame, base=1):
    """Relabel single frame sequentially
    """
    out = np.zeros(frame.shape)
    props = measure.regionprops(frame)
    for i in range(len(props)):
        lb = props[i].label
        out[frame==lb] = i + base
    
    return out, len(props)


def register_label_to_table(frame, table):
    """Register labels to the table according to centroid localization
        WARNING: will round location to 2 decimal.
    """
    tb = measure.regionprops_table(frame, properties=('centroid', 'label'))
    tb = pd.DataFrame(tb)
    tb.columns = ['Center_of_the_object_1', 'Center_of_the_object_0', 'label']
    tb['Center_of_the_object_1'] = np.round(tb['Center_of_the_object_1'], 2)
    tb['Center_of_the_object_0'] = np.round(tb['Center_of_the_object_0'], 2)
    table['Center_of_the_object_0'] = np.round(table['Center_of_the_object_0'], 2)
    table['Center_of_the_object_1'] = np.round(table['Center_of_the_object_1'], 2)

    out = pd.merge(table, tb, on=['Center_of_the_object_0', 'Center_of_the_object_1'])
    out['continuous_label'] = out['label']
    del out['label']
    missing = table.shape[0] - out.shape[0]
    return out, missing


def resolve_joined(frame, table, n=4):
    """Resolve joined frame and table of single slice

    Args:
        frame (numpy.array): joined image slice
        table (pandas.DataFrame): object table with coordinate adjusted from joining
        n (int): tile count

    Returns:
        (numpy.array): relabeled slice with objects at the edge joined
        (pandas.DataFrame): resolved object with object labels updated. 
            Objects at the edge will be deleted, new objects due to joining will be registered.
            Cell cycle phase information (prediction class and confidence) is drawn from the object
            of larger size.

    """

    if n not in [4,9]:
        raise ValueError('Join tile number should either be 4 or 9.')
    
    if frame.shape[0] % n != 0:
        raise ValueError('Stack size is not multiple of tile count n.')
    
    table = register_label_to_table(frame, table)[0]

    n = int(np.sqrt(n))
    width = int(frame.shape[0] / n)
    flooded = measure.label(frame.astype('bool'))
    new_table = pd.DataFrame()
    new_frame = np.zeros(frame.shape)
    obj_count = 1

    for i in range(n):
        for j in range(n):
            col_low = j*width
            row_low = i*width
            col_bd = (j+1)*width - 1
            row_bd = (i+1)*width - 1
            props = measure.regionprops(frame)

            for p in props:
                x, y = np.round(p.centroid, 2)
                if x < row_bd and x >= row_low and y < col_bd and y >= col_low:
                    row = table[(table['Center_of_the_object_0']==y) & (table['Center_of_the_object_1']==x)]
                    if row.shape[0]==0:
                        continue
                    ct = 0
                    for lc in p.bbox:
                        if lc == col_bd or lc == row_bd:
                            ct += 1
                    if ct >= 2:
                        break
                        # boundary object found
                        fld_label = int(np.max(flooded[p.centroid]))
                        new_frame[flooded==fld_label] = obj_count
                        # deduce object information
                        temp_mask = np.zeros(frame.shape)
                        temp_mask[flooded==fld_label] = 1
                        measure.regionprops_table(temp_mask, properties=('bbox', 'centroid'))

                        # remove sealed objects to block further recording
                        frame[flooded==fld_label] = 0

                        obj_count += 1

                    else:
                        row.loc[:]['continuous_label'] = obj_count
                        new_table = new_table.append(row)
                        new_frame[frame==p.label] = obj_count
                        obj_count += 1

    return