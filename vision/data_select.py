import os
import shutil
import glob
import zipfile
import argparse

# python data_select.py --temp temp --whole whole --current ./
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--temp", type=str, default="temp", help='path to save temporary zip files')
ap.add_argument("-w", "--whole", type=str, default="whole", help="path to save whole train files")
ap.add_argument("-c", "--current", type=str, default="./", help="path where zip files are saved")
args = vars(ap.parse_args())

def main():
    # sl, ll, el, cl을 조합하여 우리가 추출할 이미지 데이터 path를 지정해줌
    sl = ['S001', 'S002']
    ll = ['L1', 'L2', 'L3', 'L8', 'L9', 'L12', 'L22', 'L23']
    el = ['E01', 'E02']
    cl = ['C' + str(i) + '.jpg' for i in range(4, 11)] + ['C19.jpg', 'C20.jpg']
    
    # zip 파일을 풀면 그냥 생짜로 S001~S006 디렉터리를 풀어버려서 임시로 보관할 디렉터리가 필요해서
    # temp 디렉터리 안에 압축을 해제함
    # whole은 추출한 이미지 데이터만 저장하는 디렉터리
    basepath = args['temp']
    targetpath = args['whole']

    # directory 변수에 zipfile이 저장된 디렉터리로 지정해주고
    # 데이터셋 zipfile말고 다른 zipfile은 없개끔!!!!!!!!!!!!
    # 저처럼 그냥 zipfile이 저장된 디렉터리에서 실행시킬거면 directory 안바꾸셔도 되요
    directory = args['current']
    os.chdir(directory)
    
    # temp 디렉터리 만들어주기
    if basepath not in os.listdir():
        os.mkdir(basepath)
    
    # whole 디렉터리 만들어주기
    if targetpath not in os.listdir():
        os.mkdir(targetpath)
    
    # the number of zip files
    zipcount = len(glob.glob1(directory,"*.zip"))

    # 디렉터리 내에 있는 모든 zip 파일을
    for i, zf in enumerate(glob.glob('*.zip'), 1):
        pathlist = []
        zipname = zf.split('.')[0]  # zip 파일 이름 
        
        # zip 파일을 temp 디렉터리에 압축 해제해줌
        zf = zipfile.ZipFile(zf)
        zf.extractall(os.path.join(basepath))
        
        
        # (추출한 이미지 파일의 경로, 새롭게 지정할 이미지 파일의 이름) 형식으로 pathlist 변수에 저장함
        for s in sl:
            for l in ll:
                for e in el:
                    for c in cl:
                        pathlist.append((os.path.join(s, l, e, c), '_'.join([zipname, s, l, e, c])))
                        
        # temp 디렉터리에 저장한 추출한 이미지 파일을 whole 디렉터리에 옮겨줌        
        for path, pname in pathlist:
            shutil.move(os.path.join(basepath, path), os.path.join(targetpath, pname))
            # print(path, pname)
        
        print(f"Selected pre-defined image files from {zipname}.zip({i}/{zipcount})")

        # temp 디렉터리에 있는 쓰지 않는 디렉터리 및 파일 제거
        for dir in os.listdir(basepath):
            dpath = os.path.join(basepath, dir)
            if os.path.isdir(dpath):
                shutil.rmtree(dpath)
    
    # zip 파일을 전부 다 압축해제하면 temp 파일도 지워주기
    shutil.rmtree(basepath)

if __name__ == '__main__':
    main()