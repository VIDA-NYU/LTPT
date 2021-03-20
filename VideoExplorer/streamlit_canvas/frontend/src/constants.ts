interface PointConfig {
    name: string,
    coords: Array<number>,
    clickable: boolean
}
interface CanvasPoint {
    name: string,
    coords: Array<number>
}

interface lineConfig {
    src: string,
    dest: string,
    type: string
}

interface ShapeConfig {
    name: string,
    type: string,
    coords: Array<number>,
}

const pointsSource = {
    'Nose': 0,
    'RShoulder': 1,
    'RElbow': 2,
    'RWrist': 3,
    'LShoulder': 4,
    'LElbow': 5,
    'LWrist': 6,
    'RHip': 7,
    'RKnee': 8,
    'RAnkle': 9,
    'LHip': 10,
    'LKnee': 11,
    'LAnkle': 13,
    'REye': 13,
    'LEye': 14,
    'REar': 15,
    'LEar': 16,
}

const keypointConfigs: PointConfig[] = [
    {
        name: "Nose",
        coords: [0.5, 0.115],
        clickable: false
    },
    // {
    //     name: "Neck",
    //     coords: [0.5, 0.24]
    // },
    {
        name: "RShoulder",
        coords: [0.6, 0.25],
        clickable:true
    },
    {
        name: "LShoulder",
        coords: [0.4, 0.25],
        clickable: true
    },
    {
        name: "RElbow",
        coords: [0.7, 0.33],
        clickable: true
    },
    {
        name: "LElbow",
        coords: [0.3, 0.33],
        clickable: true
    },
    {
        name: "RWrist",
        coords: [0.7, 0.48],
        clickable: true
    },
    {
        name: "LWrist" ,
        coords: [0.3, 0.48],
        clickable: true
    },
    {
        name: "RHip",
        coords: [0.54, 0.5],
        clickable: true
    },
    {
        name: "LHip",
        coords: [0.46, 0.5],
        clickable: true
    },
    {
        name: "RKnee",
        coords: [0.6, 0.7],
        clickable: true
    },
    {
        name: "LKnee",
        coords: [0.4, 0.7],
        clickable: true
    },
    {
        name: "RAnkle",
        coords: [0.6, 0.9],
        clickable: true
    },
    {
        name: "LAnkle",
        coords: [0.4, 0.9],
        clickable: true
    },
    // {
    //     name: "REye",
    //     coords: [0.54, 0.07],
    //     clickable:false
    // },
    // {
    //     name: "LEye",
    //     coords: [0.46, 0.07],
    //     clickable:false
    // },
    {
        name: "REar",
        coords: [0.6, 0.1],
        clickable:false
    },
    {
        name: "LEar",
        coords: [0.4, 0.1],
        clickable:false
    }

]

let auxiliaryKeypointConfigs: Array<PointConfig> = [
    {
        name: "downSideHead",
        coords: [0.5, 0.2],
        clickable: false
    },
    {
        name: "centerHip",
        coords: [0.5, 0.5],
        clickable: false
    },
    {
        name: "downSideNeck",
        coords: [0.50,.26],
        clickable: false
    }

]

let decorationShapeConfigs: Array<ShapeConfig> = [
    {
        name: "mouth",
        type: "mouth",
        coords: [0.5, 0.14]
    },
    {
        name: "head",
        type: "head",
        coords: [0.5, 0.11]
    },
    {
        name: "LEye",
        type: "eye",
        coords: [0.46, 0.10]
    },
    {
        name: "LEye",
        type: "eye",
        coords: [0.54, 0.10]
    }
]

let auxiliaryLineConfigs = [
    {
        src: "downSideHead",
        dest: "downSideNeck",
        type: "auxiliary"
    },
    {
        src: "downSideNeck",
        dest: "RShoulder",
        type: "auxiliary"
    },
    {
        src: "RShoulder",
        dest: "RElbow",
        type: "auxiliary"
    },
    {
        src: "RElbow",
        dest: "RWrist",
        type: "auxiliary"
    },
    {
        src: "downSideNeck",
        dest: "LShoulder",
        type: "auxiliary"
    },
    {
        src: "LShoulder",
        dest: "LElbow",
        type: "auxiliary"
    },
    {
        src: "LElbow",
        dest: "LWrist",
        type: "auxiliary"
    },
    {
        src: "downSideNeck",
        dest: "centerHip",
        type: "auxiliary"
    },
    {
        src: "centerHip",
        dest: "RHip",
        type: "auxiliary"
    },
    {
        src: "RHip",
        dest: "RKnee",
        type: "auxiliary"
    },
    {
        src: "RKnee",
        dest: "RAnkle",
        type: "auxiliary"
    },
    {
        src: "centerHip",
        dest: "LHip",
        type: "auxiliary"
    },
    {
        src: "LHip",
        dest: "LKnee",
        type: "auxiliary"
    },
    {
        src: "LKnee",
        dest: "LAnkle",
        type: "auxiliary"
    },




]

let sampleData = [[0.18981481481481483, 0.3967391304347826], [0.1574074074074074, 0.44021739130434784], [0.1574074074074074, 0.44565217391304346], [0.17592592592592593, 0.532608695652174], [0.2175925925925926, 0.5543478260869565], [0.1527777777777778, 0.43478260869565216], [0.1712962962962963, 0.5217391304347826], [0.19907407407407407, 0.5380434782608695], [0.12962962962962962, 0.6032608695652174], [0.21296296296296297, 0.6032608695652174], [0.1388888888888889, 0.6902173913043478], [0.125, 0.6032608695652174], [0.18981481481481483, 0.5978260869565217], [0.12962962962962962, 0.6793478260869565], [0.18055555555555555, 0.391304347826087], [0.0, 0.0], [0.16666666666666666, 0.3967391304347826], [0.0, 0.0]]

export {pointsSource, keypointConfigs, auxiliaryKeypointConfigs, auxiliaryLineConfigs, decorationShapeConfigs}
export type { PointConfig, CanvasPoint, ShapeConfig }
