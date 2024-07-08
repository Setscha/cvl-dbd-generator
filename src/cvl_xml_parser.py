from lxml import etree as ET


def get_text_bounding_boxes(filename) -> list[list[tuple[int, int]]]:
    with open(filename, errors='ignore') as f:
        data = f.read()
        root = ET.fromstring(bytes(data, encoding='utf-16'))
        bounding_boxes = list()
        for region in root.findall('.//{*}AttrRegion[@attrType="3"]'):
            bounding_box = list()
            for point in region.findall('./{*}minAreaRect/{*}Point'):
                bounding_box.append((int(point.get('x')), int(point.get('y'))))
            bounding_boxes.append(sorted(bounding_box))
    return bounding_boxes
