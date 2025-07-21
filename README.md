# Label Studio Grounding‑DINO‑SAM Integration

> A step‑by‑step guide to setting up and running our Grounding‑DINO‑SAM backend in Label Studio.

---

## 📖 Original README

For upstream documentation and advanced configuration options, see:

[HumanSignal/label-studio-ml-backend · README.md](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#what-is-the-label-studio-ml-backend)

---

## 🛠️ Prerequisites

* **Python 3.7+**
* **Docker** & **Docker Compose** (ensure your user is in the `docker` group)

---

## 🚀 Installation/Running

1. **Install Label Studio**

   ```bash
   pip install label-studio
   ```
2. **Build & run the ML Backend**

   Use the following command to start serving the ML backend at `http://localhost:9090`:
   
   ```bash
   cd label-studio-ml-backend/label_studio_ml/examples/{MODEL_NAME}
   docker compose up
   ```


Replace `{MODEL_NAME}` with the name of the model you want to use (see [here](https://github.com/HumanSignal/label-studio-ml-backend?tab=readme-ov-file#models)). 

In most cases, you will need to set `LABEL_STUDIO_URL` and `LABEL_STUDIO_API_KEY` environment variables to allow the ML backend access to the media data in Label Studio.
[Read more in the documentation](https://labelstud.io/guide/ml#Allow-the-ML-backend-to-access-Label-Studio-data). - **Not necessary**

**Warning:** Currently, ML backends support only Legacy Tokens and do not support Personal Tokens. You will encounter an `Unauthorized Error` if you use Personal Tokens.


3. **Start the Label Studio web server**

  ```bash
  label-studio start
  ```

---

## ⚙️ Project Configuration

1. **Log in** to Label Studio with your credentials.

2. **Create a new project**:

   * Navigate to **Projects** → **Create New Project**

3. **Import data** and choose **Labeling setup** → **Custom template**.

   <details>
   <summary>Example XML Template</summary>

   ```xml
    <View>
      <!-- your image -->
      <Image name="image" value="$image" zoom="true"/>
    
      <!-- single prompt text area -->
      <TextArea name="prompt" toName="image"
                editable="true"
                rows="2" maxSubmissions="1"
                showSubmitButton="true"/>
    
      <!-- rectangle labels -->
      <RectangleLabels name="label1" toName="image">
        <Label value="fire"             background="#FFA39E"/>
        <Label value="smoke plumes"     background="#D4380D"/>
        <Label value="hot embers"       background="#FFC069"/>
        <Label value="smoldering zones" background="#AD8B00"/>
        <Label value="tree trunks"      background="#D3F261"/>
        <Label value="industrial tank"  background="#389E0D"/>
        <Label value="industrial pipe"  background="#5CDBD3"/>
        <Label value="fences"           background="#096DD9"/>
        <Label value="debris"           background="#ADC6FF"/>
        <Label value="navigable road"   background="#9254DE"/>
        <Label value="thermal hotspots" background="#F759AB"/>
        <Label value="flooded road"     background="#FFA39E"/>
        <Label value="submerged road surface" background="#D4380D"/>
        <Label value="flood entry-points"      background="#FFC069"/>
        <Label value="drain inlets"            background="#AD8B00"/>
        <Label value="chemical leaks"          background="#D3F261"/>
        <Label value="collapsed rubble"        background="#389E0D"/>
        <Label value="damaged buildings"       background="#5CDBD3"/>
        <Label value="cracked ground"          background="#096DD9"/>
        <Label value="human"                 background="#ADC6FF"/>
        <Label value="emergency personnel"     background="#9254DE"/>
        <Label value="firetrucks"              background="#F759AB"/>
        <Label value="ambulances"              background="#FFA39E"/>
        <Label value="hazard tape"             background="#D4380D"/>
        <Label value="cones"                   background="#FFC069"/>
      </RectangleLabels>
    
      <!-- brush labels -->
      <BrushLabels name="label2" toName="image">
        <Label value="fire"             background="#FFA39E"/>
        <Label value="smoke plumes"     background="#D4380D"/>
        <Label value="hot embers"       background="#FFC069"/>
        <Label value="smoldering zones" background="#AD8B00"/>
        <Label value="tree trunks"      background="#D3F261"/>
        <Label value="industrial tank"  background="#389E0D"/>
        <Label value="industrial pipe"  background="#5CDBD3"/>
        <Label value="fences"           background="#096DD9"/>
        <Label value="debris"           background="#ADC6FF"/>
        <Label value="navigable road"   background="#9254DE"/>
        <Label value="thermal hotspots" background="#F759AB"/>
        <Label value="flooded road"     background="#FFA39E"/>
        <Label value="submerged road surface" background="#D4380D"/>
        <Label value="flood entry-points"      background="#FFC069"/>
        <Label value="drain inlets"            background="#AD8B00"/>
        <Label value="chemical leaks"          background="#D3F261"/>
        <Label value="collapsed rubble"        background="#389E0D"/>
        <Label value="damaged buildings"       background="#5CDBD3"/>
        <Label value="cracked ground"          background="#096DD9"/>
        <Label value="human"                 background="#ADC6FF"/>
        <Label value="emergency personnel"     background="#9254DE"/>
        <Label value="firetrucks"              background="#F759AB"/>
        <Label value="ambulances"              background="#FFA39E"/>
        <Label value="hazard tape"             background="#D4380D"/>
        <Label value="cones"                   background="#FFC069"/>
      </BrushLabels>
    </View>
   ```

   </details>

4. **Connect the ML backend and define the Project Prompt for Automatic Pre-Labeling**:

   * Go to **Project Settings** → **Model**
   * Click **Connect Model**
   * Enter **URL:** `http://localhost:9090`
   * Enable **Interactive pre‑annotations**
   * Enter the prompt in json format

   ![Custom prompt](https://github.com/user-attachments/assets/22ea1be5-c646-4b6b-b1f0-fdb6b8150f23)


```json
{
 "prompt": "enter prompt here"
}
```



6. **Enable Auto‑annotation** on any image:

   * Open an image in the labeling interface
   * Toggle **Auto‑annotation** at the bottom
   * (Optional) Add or modify the prompt in the text area for custom queries and pick the corresponding label from below

   ![Auto Annotation](https://github.com/user-attachments/assets/b9a946b4-2a42-426b-8da5-8f9d7be8d360)

   ![Custom Prompt](https://github.com/user-attachments/assets/729d4401-f75b-4284-a4aa-20fc5d0c980f)

---

## 🛠️ Troubleshooting & Tips

* To update label colors or values, edit the `<Label>` entries in the template.
* Check container health:

  ```bash
  docker ps | grep grounding_sam
  ```
  
* In case label-studio won't load this usually does the trick, also refresh port-forwarding (via VS Code Port tab)

  ```bash
  pkill -f label-studio
  ```
