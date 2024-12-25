import { app } from "/scripts/app.js";


const CONVERTED_TYPE = 'converted-widget'
const VALID_TYPES = [
  'STRING',
  'combo',
  'number',
  'toggle',
  'BOOLEAN',
  'text',
  'string'
]
const REGIONS = [ 'skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip' ]
const DEBUGITEMS = [ 'bounding-box', 'face-landmark-5', 'face-landmark-5/68', 'face-landmark-68', 'face-landmark-68/5', 'face-mask', 'face-detector-score', 'face-landmarker-score', 'age', 'gender', 'race' ]


function hideWidget(node, widget, suffix = '') {
  if (widget.type?.startsWith(CONVERTED_TYPE)) return
  widget.origType = widget.type
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4] // -4 is due to the gap litegraph adds between widgets automatically
  widget.type = CONVERTED_TYPE + suffix
  widget.serializeValue = () => {
    // Prevent serializing the widget if we have no input linked
    if (!node.inputs) {
      return undefined
    }
    let node_input = node.inputs.find((i) => i.widget?.name === widget.name)

    if (!node_input || !node_input.link) {
      return undefined
    }
    return widget.origSerializeValue
      ? widget.origSerializeValue()
      : widget.value
  }

}

function showWidget(node,widget) {
  if (!widget.type?.startsWith(CONVERTED_TYPE)) return
  widget.type = widget.origType
  widget.computeSize = widget.origComputeSize
  widget.serializeValue = widget.origSerializeValue

  delete widget.origType
  delete widget.origComputeSize
  delete widget.origSerializeValue

}

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}


app.registerExtension({
    name: "WDTRIP",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("WDTRIP")) {
            return;
        }
        nodeType.prototype.onNodeCreated = function () {
            this.inputs_offset = 0;
            // face_mask_types
            const key_w=this.widgets.find((w) => w.name === "face_mask_types");
            if(key_w["value"]!=="region"){
                for(let i in REGIONS){
                    const target_w=this.widgets.find((w) => w.name === REGIONS[i]);
                    hideWidget(this,target_w);
                }

            };
            chainCallback(key_w, "callback", (value) => {
            if(value==="region"){
                for(let i in REGIONS){
                    const target_w=this.widgets.find((w) => w.name === REGIONS[i]);
                    showWidget(this,target_w);
                }
            }
            else{
                for(let i in REGIONS){
                    const target_w=this.widgets.find((w) => w.name === REGIONS[i]);
                    hideWidget(this,target_w);
                }
            };
          });

          // face_debug
          const key_w2=this.widgets.find((w) => w.name === "face_debug");
            if(!key_w2["value"]){
                for(let i in DEBUGITEMS){
                    const target_w=this.widgets.find((w) => w.name === DEBUGITEMS[i]);
                    hideWidget(this,target_w);
                }

            };
            chainCallback(key_w2, "callback", (value) => {
            if(key_w2["value"]){
                for(let i in DEBUGITEMS){
                    const target_w=this.widgets.find((w) => w.name === DEBUGITEMS[i]);
                    showWidget(this,target_w);
                }
            }
            else{
                for(let i in DEBUGITEMS){
                    const target_w=this.widgets.find((w) => w.name === DEBUGITEMS[i]);
                    hideWidget(this,target_w);
                }
            };
          });

        };
//        switch (nodeData.name) {
//            case "hxy":
//                break;
//        }
    }
});
