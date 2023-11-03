import numpy as np
import cv2 as cv
import glob
import argparse
import os


def main():
    # 接收字符
    parser = argparse.ArgumentParser(description="caculate camera-calibration parameters", add_help=False)
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")
    parser.add_argument("-r", "--rows", help="number of detect connor in rows", default="0", action="store", dest="rows", type=int)
    parser.add_argument("-c", "--columns", help="number of detect connor in columns", default="0", action="store", dest="columns", type=int)
    parser.add_argument("-input", "--input", help="input pictures path", default="input", action="store", dest="input")
    args = parser.parse_args()

    show_help = args.show_help
    if show_help:
        parser.print_help()
        return
    # 判断是否输入必须参数
    if args.rows == 0 or args.columns == 0:
        raise ValueError(
                "Please input the number of connor in rows or colums that you want to detect on your chessboard"
            )

    columns = args.columns
    rows = args.rows
    input_path = os.getcwd() + "\\" + args.input
    if not os.path.exists(input_path):
        raise ValueError(
                "Please input the right input path"
            )        

    # 输出路径
    output_path = os.getcwd() + "\output"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    print(" the result and process picture is restore in: " + output_path)

    # 标准设置
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 根据说需要测试的棋盘角个数生成相应坐标
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:columns,0:rows].T.reshape(-1,2)

    objpoints = [] 
    imgpoints = [] 
    images = glob.glob(input_path + "\\" +'*.jpg')

    # 检测input文件夹中是否有图片
    if images == []:
        raise ValueError(
                "the input floder has no pictures"
            )        
    fail_picture = []
    # 依此处理检测到的图片  
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, (rows,columns), None)
 
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            combined = np.concatenate([cv.imread(fname), cv.drawChessboardCorners(img, (rows,columns), corners2, ret)], axis=1)
            cv.imwrite( output_path + "\\" + fname.split("\\")[-1].split(".")[0] + '_afterprocess' + '.png', combined)
            print( "---" + fname.split("\\")[-1].split(".")[0] + " has benn already processed" + "---" )
        else:
            fail_picture.append(fname.split("\\")[-1].split(".")[0])
            print( "!!! " + fname.split("\\")[-1].split(".")[0] + " is no picture of chessboard or you set wrong input rows or colunms" + " !!!" )
        
    if objpoints == []:
        raise ValueError(
                "all of the input pictures can't be detected as chessboard"
            )             
    
    print("----all input pictures have been processsed----")

    # 对相机内参进行计算
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # 输出计算后的相机内参
    print(" ")
    print("the intrinsic parameter: ")
    print("     ret:{} " .format(ret))
    print("     cameraMatrix:{}"  .format(mtx))
    print("     distortion coefficients:{}" .format(dist))
    print("     vector of rotation vectors estimated for each pattern view:{}" .format(rvecs))
    print("     vector of translation vectors estimated for each pattern view{}:" .format(tvecs)) 
 
    # 估计计算出的相机内参的准确性：越接近0，计算出的数据越准确
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )    

    # 根据计算出的内参，优化输入的图片，并将其保存在该文件夹的output文件夹下
    for fname in images:
        if fname.split("\\")[-1].split(".")[0] not in fail_picture:
            img = cv.imread(fname)
        
            h = img.shape[0]
            w = img.shape[1]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
            dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            img = img[y:y+h, x:x+w]
            combined = np.concatenate([img, dst], axis=1)
            cv.imwrite(output_path + "\\" + fname.split("\\")[-1].split(".")[0]  + '_optimized' + '.png', combined)

    print("the process is done successfully")



if __name__ == "__main__":
    main()