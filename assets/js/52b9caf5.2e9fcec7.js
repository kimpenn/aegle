"use strict";(self.webpackChunkaegle_docs=self.webpackChunkaegle_docs||[]).push([[550],{4439:(e,s,n)=>{n.r(s),n.d(s,{assets:()=>a,contentTitle:()=>o,default:()=>u,frontMatter:()=>l,metadata:()=>i,toc:()=>c});const i=JSON.parse('{"id":"SamplePreprocess/Outputs","title":"Outputs","description":"After running this preprocessing module, you will have:","source":"@site/docs/SamplePreprocess/Outputs.md","sourceDirName":"SamplePreprocess","slug":"/SamplePreprocess/Outputs","permalink":"/aegle/docs/SamplePreprocess/Outputs","draft":false,"unlisted":false,"editUrl":"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/SamplePreprocess/Outputs.md","tags":[],"version":"current","sidebarPosition":1,"frontMatter":{"sidebar_position":1},"sidebar":"tutorialSidebar","previous":{"title":"Sample Preprocess","permalink":"/aegle/docs/category/sample-preprocess"},"next":{"title":"Sample Preprocessing Overview","permalink":"/aegle/docs/SamplePreprocess/Overview"}}');var r=n(4848),t=n(8453);const l={sidebar_position:1},o="Outputs",a={},c=[{value:"Tissue Region Files",id:"tissue-region-files",level:2},{value:"Antibody Data",id:"antibody-data",level:2},{value:"Log Files",id:"log-files",level:2}];function d(e){const s={code:"code",h1:"h1",h2:"h2",header:"header",li:"li",p:"p",strong:"strong",ul:"ul",...(0,t.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(s.header,{children:(0,r.jsx)(s.h1,{id:"outputs",children:"Outputs"})}),"\n",(0,r.jsx)(s.p,{children:"After running this preprocessing module, you will have:"}),"\n",(0,r.jsx)(s.h2,{id:"tissue-region-files",children:"Tissue Region Files"}),"\n",(0,r.jsxs)(s.ul,{children:["\n",(0,r.jsxs)(s.li,{children:["\n",(0,r.jsxs)(s.p,{children:[(0,r.jsx)(s.strong,{children:"OME-TIFF tissue crops"}),": ",(0,r.jsx)(s.code,{children:"{OUT_DIR}/{EXP_ID}/{base_name}_tissue_{i}.ome.tiff"})]}),"\n",(0,r.jsxs)(s.ul,{children:["\n",(0,r.jsx)(s.li,{children:"Each crop is a multi-channel image representing a distinct tissue region"}),"\n",(0,r.jsx)(s.li,{children:"Images are saved in OME-TIFF format (C,H,W) with LZW compression"}),"\n",(0,r.jsxs)(s.li,{children:["If ",(0,r.jsx)(s.code,{children:"skip_roi_crop"})," is enabled, a single full image will be saved as ",(0,r.jsx)(s.code,{children:"{base_name}_full.ome.tiff"})]}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(s.li,{children:["\n",(0,r.jsxs)(s.p,{children:[(0,r.jsx)(s.strong,{children:"Visualization (optional)"}),": ",(0,r.jsx)(s.code,{children:"{OUT_DIR}/{EXP_ID}/tissue_masks_preview.png"})]}),"\n",(0,r.jsxs)(s.ul,{children:["\n",(0,r.jsxs)(s.li,{children:["Visual representation of detected tissue regions if ",(0,r.jsx)(s.code,{children:"visualize"})," is enabled"]}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(s.h2,{id:"antibody-data",children:"Antibody Data"}),"\n",(0,r.jsxs)(s.ul,{children:["\n",(0,r.jsxs)(s.li,{children:["\n",(0,r.jsxs)(s.p,{children:[(0,r.jsx)(s.strong,{children:"Antibody mapping"}),": ",(0,r.jsx)(s.code,{children:"{OUT_DIR}/extras/antibodies.tsv"})]}),"\n",(0,r.jsxs)(s.ul,{children:["\n",(0,r.jsxs)(s.li,{children:["Tab-separated file with columns: ",(0,r.jsx)(s.code,{children:"version"}),", ",(0,r.jsx)(s.code,{children:"channel_id"}),", and ",(0,r.jsx)(s.code,{children:"antibody_name"})]}),"\n",(0,r.jsx)(s.li,{children:"Maps each channel to its corresponding antibody marker"}),"\n"]}),"\n"]}),"\n",(0,r.jsxs)(s.li,{children:["\n",(0,r.jsxs)(s.p,{children:[(0,r.jsx)(s.strong,{children:"OME-XML metadata"}),": ",(0,r.jsx)(s.code,{children:"{data_path}/{base_name}.xml"})]}),"\n",(0,r.jsxs)(s.ul,{children:["\n",(0,r.jsx)(s.li,{children:"Extracted OME-XML metadata from the original QPTIFF file"}),"\n",(0,r.jsx)(s.li,{children:"Contains detailed information about channels, dimensions, etc."}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,r.jsx)(s.h2,{id:"log-files",children:"Log Files"}),"\n",(0,r.jsxs)(s.ul,{children:["\n",(0,r.jsxs)(s.li,{children:["Detailed logs for each experiment are saved to ",(0,r.jsx)(s.code,{children:"{LOG_DIR}/{EXP_ID}.log"})]}),"\n",(0,r.jsx)(s.li,{children:"Contains timing information, processing steps, and any errors/warnings"}),"\n"]})]})}function u(e={}){const{wrapper:s}={...(0,t.R)(),...e.components};return s?(0,r.jsx)(s,{...e,children:(0,r.jsx)(d,{...e})}):d(e)}},8453:(e,s,n)=>{n.d(s,{R:()=>l,x:()=>o});var i=n(6540);const r={},t=i.createContext(r);function l(e){const s=i.useContext(t);return i.useMemo((function(){return"function"==typeof e?e(s):{...s,...e}}),[s,e])}function o(e){let s;return s=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:l(e.components),i.createElement(t.Provider,{value:s},e.children)}}}]);