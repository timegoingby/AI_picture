import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
# 非常推荐使用这个subprocess.Popen("pwd")执行shell指令
import subprocess

# from infer import inferEngine

# inferObject = inferEngine()

backgroudDict = {"红色": "red",
                 "蓝色": "blue",
                 "白色": "white"}

# materialDict = {
#     "脸颊素材1":"cheek_1",
#     "脸颊素材2":"cheek_2",
#     "眼镜素材1":"eye_1",
#     "眼镜素材2":"eye_2",
#     "头饰素材1":"head_1",
# }

# 标题栏
# st.markdown("<center><h1>证件照一键升恒</h1></center>",unsafe_allow_html=True)
st.image(
    "https://ai-studio-static-online.cdn.bcebos.com/3c76da448c154e72883f74091e6f8c2fdca3ad6e1de44a619c838b22fa7cfcb2")
st.markdown(
    "<center>🧒 <a style='background: linear-gradient(to right, red, blue);-webkit-background-clip: text;color: transparent;'>还  在  为  证  件  照  发  愁  吗？</a> </center>",
    unsafe_allow_html=True)
st.markdown(
    "<center>🌟 <a style='background: linear-gradient(to right, red, blue);-webkit-background-clip: text;color: transparent;'>手  机  拍  照  生  成  证  件  照！</a> </center>",
    unsafe_allow_html=True)
st.markdown(
    "<center>🌟 <a style='background: linear-gradient(to right, red, blue);-webkit-background-clip: text;color: transparent;'>1  寸  2  寸  抠  图  背  景  换  色，</a> </center>",
    unsafe_allow_html=True)
st.markdown(
    "<center>🌟 <a style='background: linear-gradient(to right, red, blue);-webkit-background-clip: text;color: transparent;'>混  排  5  寸  6  寸  照  片  冲  洗！</a> </center>",
    unsafe_allow_html=True)
st.markdown("<hr />", unsafe_allow_html=True)

# 下载库
# subprocess.Popen("wget https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-1024.tgz && tar -xvf PP-Matting-1024.tgz")
import os

# os.system("wget https://bj.bcebos.com/paddlehub/fastdeploy/PP-Matting-1024.tgz && tar -xvf PP-Matting-1024.tgz")

# 图片上传
placeholder_1 = st.empty()
with placeholder_1.container():
    st.info('点击下面上传一张自拍图片', icon="1️⃣")
    per_image = st.file_uploader(label="上传图片", type=['png', 'jpg', 'jpeg'], label_visibility='hidden')

# 图片展示
placeholder_2 = st.empty()
with placeholder_2.container():
    if per_image:
        # placeholder_1.empty()
        st.image(per_image)
        cur_img = Image.open(per_image)
        cur_img.save("temp.png")
    else:
        # st.image(
        #     "https://ai-studio-static-online.cdn.bcebos.com/43674b34618f4cde813a5d1bb895e5d627f974d2d1f0486a806ee83f971ee326")
        cur_img = Image.open("beautysmall.jpg")
        cur_img.save("temp.png")

# 选择项
placeholder_3 = st.empty()
with placeholder_3.container():
    st.info('选择证件照的背景色', icon="2️⃣")
    with st.form("summit_form"):
        backgroundOption = st.selectbox('选择你想要的背景色 🌅', [k for k in backgroudDict])
        # materialOption = st.selectbox('选择一个装饰素材 🧑‍🎄', [m for m in materialDict])
        st.info('最后一步，点击提交吧', icon="3️⃣")
        summit_btn = st.form_submit_button("提交图片")

# 图片生成
if summit_btn:
    placeholder_2.empty()
    placeholder_3.empty()
    placeholder_4 = st.empty()
    with st.spinner('正在为您生成图片，请稍后...'):
        with placeholder_4.container():
            st.warning('耐心等待，不要关闭页面噢～', icon="⚠️")
        # per_image = Image.open(per_image).convert("RGB")
        # per_image = cv2.cvtColor(np.asarray(per_image), cv2.COLOR_RGB2BGR)
        # imgResult = inferObject.run(img=per_image, backgroundName=backgroudDict[backgroundOption], material=materialDict[materialOption], option=materialDict[materialOption].split("_")[0])
        import fastdeploy as fd
        import cv2
        import os

        # 配置runtime，加载模型
        # runtime_option = build_option(args)
        runtime_option = None
        argsmodel = "./PP-Matting-1024/"
        model_file = os.path.join(argsmodel, "model.pdmodel")
        params_file = os.path.join(argsmodel, "model.pdiparams")
        config_file = os.path.join(argsmodel, "deploy.yaml")
        # model = fd.vision.matting.PPMatting(
        #     model_file, params_file, config_file, runtime_option=runtime_option)
        model = fd.vision.matting.PPMatting(
            model_file, params_file, config_file)

        # 图片抠图
        # per_image = Image.open(per_image).convert("RGB")
        # per_image = Image.open(per_image)
        # per_image = cv2.cvtColor(np.asarray(per_image), cv2.COLOR_RGB2BGR)
        im = cv2.imread("temp.png")
        bgrgb = backgroudDict[backgroundOption]
        if bgrgb == "red":
            bgrgb = cv2.imread("red.jpg")
        elif bgrgb == "blue":
            bgrgb = cv2.imread("blue.jpg")
        else:
            bgrgb = cv2.imread("white.jpg")
        bg = bgrgb
        print("*" * 20)
        result = model.predict(im.copy())
        # print(result)
        # 可视化结果
        # vis_im = fd.vision.vis_matting(im, result)
        vis_im_with_bg = fd.vision.swap_background(im, bg, result)
        cv2.imwrite("visualized_result_replaced_bg.jpg", vis_im_with_bg)

        # 输出5寸和6寸的排版打印照片
        from printpic import cut_photo
        from printpic import resize_photo

        from printpic import layout_photo_5_1
        from printpic import layout_photo_5_2
        from printpic import layout_photo_6_1
        from printpic import layout_photo_6_2
        from printpic import layout_photo_5_mix
        from printpic import layout_photo_6_mix1
        from printpic import layout_photo_6_mix2

        from PIL import Image, ImageDraw

        im = Image.open('visualized_result_replaced_bg.jpg')
        img_1 = resize_photo(cut_photo(im, 1), 1)
        img_0 = cut_photo(im, 1)
        img_51 = layout_photo_5_1(resize_photo(cut_photo(im, 1), 1))
        img_52 = layout_photo_5_2(resize_photo(cut_photo(im, 2), 2))
        img_61 = layout_photo_6_1(resize_photo(cut_photo(im, 1), 1))
        img_62 = layout_photo_6_2(resize_photo(cut_photo(im, 2), 2))
        img_512 = layout_photo_5_mix(resize_photo(cut_photo(im, 1), 1),
                                     resize_photo(cut_photo(im, 2), 2).rotate(90, expand=True))
        img_612 = layout_photo_6_mix1(resize_photo(cut_photo(im, 1), 1).rotate(90, expand=True),
                                      resize_photo(cut_photo(im, 2), 2))
        img_613 = layout_photo_6_mix2(resize_photo(cut_photo(im, 1), 1), resize_photo(cut_photo(im, 2), 2))
        # st.image(imgResult)
        st.info('生成1寸证件照295 * 413像素', icon="0️⃣")
        st.image(img_1)
        st.info('生成原精度1寸证件比例打印版', icon="1️⃣")
        st.image(img_0)
        st.info('生成5寸排版1寸证件照', icon="2️⃣")
        st.image(img_51)
        st.info('生成5寸排版2寸证件照', icon="3️⃣")
        st.image(img_52)
        st.info('生成6寸排版1寸证件照', icon="4️⃣")
        st.image(img_61)
        st.info('生成6寸排版2寸证件照', icon="5️⃣")
        st.image(img_62)
        st.info('生成5寸排版1寸2寸混合证件照', icon="6️⃣")
        st.image(img_512)
        st.info('生成6寸排版1寸2寸混合证件照', icon="7️⃣")
        st.image(img_612)
        st.info('生成6寸排版1寸2寸混合证件照', icon="8️⃣")
        st.image(img_613)
        placeholder_4.empty()
    st.success('完成啦，长按图片即可保存噢!', icon="☝")
    summit_btn = False
    # st.balloons()

# 页脚
st.markdown("<hr />", unsafe_allow_html=True)
st.markdown("<center><h6>Powered by <a style='color: lightblue'>FutureVisual</a></h6></center>",
            unsafe_allow_html=True)