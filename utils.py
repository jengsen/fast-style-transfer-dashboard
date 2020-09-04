import os
import pathlib
import json

import dash_core_components as dcc
import plotly.graph_objs as go
import dash_reusable_components as drc
import tensorflow as tf

from PIL import Image, ImageFilter, ImageDraw, ImageEnhance

#
APP_PATH = str(pathlib.Path(__file__).parent.resolve())

# [filename, image_signature, action_stack]
STORAGE_PLACEHOLDER = json.dumps(
    {"filename": None, "image_signature": None, "action_stack": []}
)

IMAGE_STRING_PLACEHOLDER = drc.pil_to_b64(
    Image.open(os.path.join(APP_PATH, os.path.join("images", "default.jpg"))).copy(),
    enc_format="jpeg",
)


GRAPH_PLACEHOLDER = dcc.Graph(
    id="interactive-image",
    figure={
        "data": [],
        "layout": {
            "autosize": True,
            "paper_bgcolor": "#272a31",
            "plot_bgcolor": "#272a31",
            "margin": go.Margin(l=40, b=40, t=26, r=10),
            "xaxis": {
                "range": (0, 1527),
                "scaleanchor": "y",
                "scaleratio": 1,
                "color": "white",
                "gridcolor": "#43454a",
                "tickwidth": 1,
            },
            "yaxis": {
                "range": (0, 1200),
                "color": "white",
                "gridcolor": "#43454a",
                "tickwidth": 1,
            },
            "images": [
                {
                    "xref": "x",
                    "yref": "y",
                    "x": 0,
                    "y": 0,
                    "yanchor": "bottom",
                    "sizing": "stretch",
                    "sizex": 1527,
                    "sizey": 1200,
                    "layer": "below",
                    "source": "/images/default.jpg",
                }
            ],
            "dragmode": "select",
        },
    },
)

# Maps process name to the Image filter corresponding to that process
FILTERS_DICT = {
    "blur": ImageFilter.BLUR,
    "contour": ImageFilter.CONTOUR,
    "detail": ImageFilter.DETAIL,
    "edge_enhance": ImageFilter.EDGE_ENHANCE,
    "edge_enhance_more": ImageFilter.EDGE_ENHANCE_MORE,
    "emboss": ImageFilter.EMBOSS,
    "find_edges": ImageFilter.FIND_EDGES,
    "sharpen": ImageFilter.SHARPEN,
    "smooth": ImageFilter.SMOOTH,
    "smooth_more": ImageFilter.SMOOTH_MORE,
}

ENHANCEMENT_DICT = {
    "color": ImageEnhance.Color,
    "contrast": ImageEnhance.Contrast,
    "brightness": ImageEnhance.Brightness,
    "sharpness": ImageEnhance.Sharpness,
}


def generate_lasso_mask(image, selectedData):
    """
    Generates a polygon mask using the given lasso coordinates
    :param selectedData: The raw coordinates selected from the data
    :return: The polygon mask generated from the given coordinate
    """

    height = image.size[1]
    y_coords = selectedData["lassoPoints"]["y"]
    y_coords_corrected = [height - coord for coord in y_coords]

    coordinates_tuple = list(zip(selectedData["lassoPoints"]["x"], y_coords_corrected))
    mask = Image.new("L", image.size)
    draw = ImageDraw.Draw(mask)
    draw.polygon(coordinates_tuple, fill=255)

    return mask


def apply_filters(image, zone, filter, mode):
    filter_selected = FILTERS_DICT[filter]

    if mode == "select":
        crop = image.crop(zone)
        crop_mod = crop.filter(filter_selected)
        image.paste(crop_mod, zone)

    elif mode == "lasso":
        im_filtered = image.filter(filter_selected)
        image.paste(im_filtered, mask=zone)


def apply_enhancements(image, zone, enhancement, enhancement_factor, mode):
    enhancement_selected = ENHANCEMENT_DICT[enhancement]
    enhancer = enhancement_selected(image)

    im_enhanced = enhancer.enhance(enhancement_factor)

    if mode == "select":
        crop = im_enhanced.crop(zone)
        image.paste(crop, box=zone)

    elif mode == "lasso":
        image.paste(im_enhanced, mask=zone)


def show_histogram(image):
    def hg_trace(name, color, hg):
        line = go.Scatter(
            x=list(range(0, 256)),
            y=hg,
            name=name,
            line=dict(color=(color)),
            mode="lines",
            showlegend=False,
        )
        fill = go.Scatter(
            x=list(range(0, 256)),
            y=hg,
            mode="lines",
            name=name,
            line=dict(color=(color)),
            fill="tozeroy",
            hoverinfo="none",
        )

        return line, fill

    hg = image.histogram()

    if image.mode == "RGBA":
        rhg = hg[0:256]
        ghg = hg[256:512]
        bhg = hg[512:768]
        ahg = hg[768:]

        data = [
            *hg_trace("Red", "#FF4136", rhg),
            *hg_trace("Green", "#2ECC40", ghg),
            *hg_trace("Blue", "#0074D9", bhg),
            *hg_trace("Alpha", "gray", ahg),
        ]

        title = "RGBA Histogram"

    elif image.mode == "RGB":
        # Returns a 768 member array with counts of R, G, B values
        rhg = hg[0:256]
        ghg = hg[256:512]
        bhg = hg[512:768]

        data = [
            *hg_trace("Red", "#FF4136", rhg),
            *hg_trace("Green", "#2ECC40", ghg),
            *hg_trace("Blue", "#0074D9", bhg),
        ]

        title = "RGB Histogram"

    else:
        data = [*hg_trace("Gray", "gray", hg)]

        title = "Grayscale Histogram"

    layout = go.Layout(
        autosize=True,
        title=title,
        margin=go.Margin(l=50, r=30),
        legend=dict(x=0, y=1.15, orientation="h"),
        paper_bgcolor="#31343a",
        plot_bgcolor="#272a31",
        font=dict(color="darkgray"),
        xaxis=dict(gridcolor="#43454a"),
        yaxis=dict(gridcolor="#43454a"),
    )

    return go.Figure(data=data, layout=layout)


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(
        input_shape[1] * input_shape[2] * input_shape[3], tf.float32
    )
    return result / num_locations


def style_loss(gram_style, style_features_transformed):
    style_loss = tf.add_n(
        [
            tf.reduce_mean((gram_matrix(sf_transformed) - gm) ** 2)
            for sf_transformed, gm in zip(
                style_features_transformed, gram_style
            )
        ]
    )
    return style_loss


def content_loss(content_features, content_features_transformed):
    content_loss = tf.add_n(
        [
            tf.reduce_mean((cf_transformed - cf) ** 2)
            for cf_transformed, cf in zip(
                content_features_transformed, content_features
            )
        ]
    )
    return content_loss

