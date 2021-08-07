using System;
using System.Collections.Generic;
using UnityEngine;
using System.Collections;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine.UIElements;

/// <summary>
/// this behavior is used to help debugging the incomming packages.
/// it creates a simple graphical representations for all rigid bodies received 
/// through the network
/// </summary>
public class SimpleDebugRB : MonoBehaviour {

    // singleton
    public static SimpleDebugRB Instance;

    [SerializeField] private Material matPrediction, matGroundTruth;
    [SerializeField] private float _skeletonSpacing = 0;
    [SerializeField] private int sampleRate; //how often (frames) to wait between sampling incomming animation
    [SerializeField] private GameObject jointTemplate = null;
    [SerializeField] private Color originalColor = Color.blue;
    [SerializeField] private Color predictionColor = Color.red;
    [SerializeField] private Color sampleTint = Color.white;
    [SerializeField] private float sampleColorStep = 0.2f;

    Dictionary<string, Dictionary<string, Transform>> skeletons = new Dictionary<string, Dictionary<string, Transform>>();

    Dictionary<string, string> _childParentMapping = new Dictionary<string, string>();
    Dictionary<Transform,Transform> _debugDrawHierarchy = new Dictionary<Transform, Transform>();
    //public GameObject skeleton;

    private List<float> _predictionTimes = new List<float>();
    private int _maxPredictionTimeCount = 1000;
    private float _minPredictionTime = 1.0f;
    private float _maxPredictionTime = 0.0f;

    private float currentSampleLength;
    private float currentNumSamples;

    //List of sampled animation frames for visualization. Pair is: (<skeleton name>,<list of joint positions>) 
    private Dictionary<string, List<Transform>> frameSamples = new Dictionary<string, List<Transform>>();
    //Pr-skeleton frame timers used for sampling every n-frames
    private Dictionary<string, int> sampleTimers = new Dictionary<string, int>();

    private void Awake()
    {
        if (Instance != null && Instance != this)
            Destroy(this.gameObject);
        else
            Instance = this;
    }

    private void OnValidate()
    {
        if (sampleRate < 0) sampleRate = 0;
    }

    void DrawLinks()
    {
        foreach(var skeleton_name in skeletons.Keys)
        {
            var skeleton = skeletons[skeleton_name];
            Color drawColor = skeleton_name == "original" ? originalColor : predictionColor;
            foreach(var joint in skeleton.Values)
            {
                if(joint.name.Trim().Equals("Hips"))
                {
                    continue;
                }
                if(joint.transform.parent != null && !joint.transform.parent.name.Equals("Hips"))
                {
                    Debug.DrawLine(joint.position, joint.transform.parent.position, drawColor);
                }
                else if(!_debugDrawHierarchy.ContainsKey(joint)) //Global space joint has no transform parent
                {
                    string parentName = _childParentMapping[joint.name];
                    if(parentName.Trim().Equals("Hips"))
                    {
                        continue;
                    }
                    GameObject parent = null;
                    while(parent == null)
                    {
                        foreach(GameObject go in GameObject.FindObjectsOfType<GameObject>())
                        {
                            if(go.name.Equals(parentName))
                            {
                                if(go.transform.parent != null && !go.transform.parent.name.Equals("Hips")) continue;
                                if(skeleton.ContainsValue(go.transform))
                                {
                                    parent = go;
                                    break;
                                }
                            }
                        }
                        parentName = _childParentMapping[parentName];
                        if(parentName.Trim().Equals("Hips"))
                        {
                            continue;
                        }
                    }
                    if(parent != null)
                    {
                        _debugDrawHierarchy.Add(joint, parent.transform);
                    }
                } else 
                {
                    Debug.DrawLine(joint.position, _debugDrawHierarchy[joint].position, drawColor);
                }
                
            } 
        }
    }

    void Update()
    {
        while(PythonLauncher.data.Count > 0)
        {
            try
            {
                HandlePythonData(PythonLauncher.data[0]);
                PythonLauncher.data.RemoveAt(0);
                
            }
            catch (System.Exception e)
            {
                Debug.LogWarning("Something went wrong with: " + PythonLauncher.data[0] + "\n" + e.Message + "\n" + e.StackTrace);
                PythonLauncher.data.RemoveAt(0);
            }
           
        }
        //DrawLinks();
    }

    float parsefloat(string s) 
    {
        return float.Parse(s, CultureInfo.InvariantCulture.NumberFormat);
    }

    public void HandlePythonData(string info)
    {
        string[] infos = info.Split(' ');
        switch(infos[0])
        {
            case "H": //Hierarchy info
                ProcessHierarchyInfo(infos);
                break;
            case "O": //Joint offset
                ProcessJointOffsets(infos);
                break;
            case "J": //Joint positions (local)(?)
                ProcessSkeletonJointData(infos);
                break;
            case "G": //Joint positions (global)
                ProcessGlobalJointData(infos);
                break;
            case "X":
                ProcessAngleAxisSkeletonJointData(infos);
                break;
            case "T":
                UpdatePredictionTime(infos);
                break;
            case "B": //Begin animation
                ProcessAnimationStart(infos);
                break;
            case "E": //Start of pose marker
                ProcessPoseStart(infos);
                break;
            default:
                throw new System.Exception("Python input not recognized!");
        }
        
    }

    void UpdatePredictionTime(string[] info)
    {
        float predictionTime = parsefloat(info[1]);
        if(_predictionTimes.Count > 50)
        {
            _maxPredictionTime = Mathf.Max(_maxPredictionTime, predictionTime);
            _minPredictionTime = Mathf.Min(_minPredictionTime, predictionTime);
        }

        _predictionTimes.Add(predictionTime);
        if(_predictionTimes.Count  > _maxPredictionTimeCount) _predictionTimes.RemoveAt(0);
    }

    void ProcessHierarchyInfo(string[] info)
    {
        string hierarchyName = info[1];
        skeletons.Add(hierarchyName, new Dictionary<string, Transform>());
        frameSamples.Add(hierarchyName, new List<Transform>()); //TODO: Fix this
        sampleTimers.Add(hierarchyName, sampleRate);
        
        GameObject hierarchyParent = new GameObject(hierarchyName);

        for(uint pairIndex = 2; pairIndex < info.Length; pairIndex++)
        {
            string[] childParent = info[pairIndex].Split('-');
            if(childParent.Length < 2) break;
            string child = childParent[0];
            string parent = childParent[1];
            if(!_childParentMapping.ContainsKey(child))
            {
                _childParentMapping.Add(child, parent);
            } 
            if(!parent.Equals("") && !skeletons[hierarchyName].ContainsKey(parent))
            {
                GameObject parentJoint = CreateJoint(parent, info[1]);
                skeletons[hierarchyName].Add(parent, parentJoint.transform);
            }
            if(!skeletons[hierarchyName].ContainsKey(child))
            {
                GameObject childJoint = CreateJoint(child, info[1]);
                if (!parent.Equals("")) childJoint.transform.parent = skeletons[hierarchyName][parent];
                else childJoint.transform.parent = hierarchyParent.transform;
                skeletons[hierarchyName].Add(child, childJoint.transform);
            }
        }
    }

    void ProcessJointOffsets(string[] info)
    {
        string hierarchyName = info[1];
        string jointName = info[2];
        skeletons[hierarchyName][jointName].localPosition = new Vector3(parsefloat(info[3]), parsefloat(info[4]), parsefloat(info[5]));
    }

    void ProcessSkeletonJointData(string[] info)
    {
        string skeletonName = info[1];
        string jointName = info[2];
        Vector3 position = new Vector3(parsefloat(info[3]), parsefloat(info[4]), parsefloat(info[5]));
        Vector3 rotation = new Vector3(parsefloat(info[6]), parsefloat(info[7]), parsefloat(info[8]));
        UpdateRigidBody(skeletonName, jointName, position, rotation);
    }

    void ProcessAngleAxisSkeletonJointData(string[] info)
    {
        string skeletonName = info[1];
        string jointName = info[2];
        Vector3 position = new Vector3(parsefloat(info[3]), parsefloat(info[4]), parsefloat(info[5]));
        Vector3 axis = new Vector3(parsefloat(info[6]), parsefloat(info[7]), parsefloat(info[8])).normalized;
        float angle = Mathf.Rad2Deg * parsefloat(info[9]);
        UpdateRigidBody_AngleAxis(skeletonName, jointName, position, axis, angle);
    }

    void ProcessGlobalJointData(string[] info)
    {
        string skeletonName = info[1];
        string jointName = info[2];
        Vector3 position = new Vector3(parsefloat(info[3]), parsefloat(info[4]), parsefloat(info[5]));
        //        Vector3 rotation = new Vector3(parsefloat(info[6]), parsefloat(info[7]), parsefloat(info[8]));
        Quaternion rotation = new Quaternion(parsefloat(info[6]), parsefloat(info[7]), parsefloat(info[8]), parsefloat(info[9]));
        if (!skeletons.ContainsKey(skeletonName))
        {
            skeletons.Add(skeletonName, new Dictionary<string, Transform>());
        }
        if(!skeletons[skeletonName].ContainsKey(jointName))
        {
            skeletons[skeletonName].Add(jointName, CreateJoint(jointName, skeletonName).transform);
        }

        if (skeletonName == "original")
        {
            position += Vector3.left * _skeletonSpacing;
        }
        
        
        UpdateRigidBody(skeletonName, jointName, position, rotation);
    }

    public void UpdateRigidBody(string skeletonName, string jointName, Vector3 position, Quaternion quat)
    {
        /*if(skeletons[skeletonName][jointName].parent == null)
        {
            skeletons[skeletonName][jointName].localPosition = position;
        }*/
        
        // skeletons[skeletonName][jointName].localPosition = position;
        // skeletons[skeletonName][jointName].position = position; //global pos
        
        skeletons[skeletonName][jointName].position = position;

        skeletons[skeletonName][jointName].localRotation = quat;

    }

    public void UpdateRigidBody(string skeletonName, string jointName, Vector3 position, Vector3 eulerAngles)
    {
        /*if(skeletons[skeletonName][jointName].parent == null)
        {
            skeletons[skeletonName][jointName].localPosition = position;
        }*/
        skeletons[skeletonName][jointName].localPosition = position;
        skeletons[skeletonName][jointName].localRotation =
                                                    Quaternion.AngleAxis(eulerAngles.z, Vector3.forward) *
                                                    Quaternion.AngleAxis(eulerAngles.y, Vector3.up) *
                                                    Quaternion.AngleAxis(eulerAngles.x, Vector3.right);
    }

    public void UpdateRigidBody_AngleAxis(string skeletonName, string jointName, Vector3 position, Vector3 axis, float angle)
    {
        if(skeletons[skeletonName][jointName].parent == null)
        {
            skeletons[skeletonName][jointName].localPosition = position;
        } 
        skeletons[skeletonName][jointName].localRotation = Quaternion.AngleAxis(angle, axis);
    }
	
	/// <summary>
	/// creates a graphical representation in the shape of a shoebox.
	/// </summary>
	/// <returns>The shoe box.</returns>
	/// <param name="scale">A scale multiplier.</param>
    GameObject CreateJoint(string name, string skeletonName="", float scale = .5f)
    {
        GameObject newGO;
        if (jointTemplate == null)
        {
            newGO = GameObject.CreatePrimitive(PrimitiveType.Sphere); // was primitiveType.Cube before
        }
        else
        {
            newGO = Instantiate(jointTemplate);
            Color col = skeletonName == "original" ? originalColor : predictionColor;

            newGO.GetComponent<DrawLinkToParent>().drawColor = col;

        }
        
        if (skeletonName.Equals("prediction") ) {
            newGO.GetComponent<Renderer>().material = matPrediction;
        }
        else
        {
            newGO.GetComponent<Renderer>().material = matGroundTruth;
        }
        newGO.transform.localScale = new Vector3(1.0f, 1.0f, 1.0f) * scale;
        newGO.name = name;

        return newGO;
    }

    void ProcessAnimationStart(string[] info)
    {
        int sampleLength = int.Parse(info[1]);
        currentSampleLength = sampleLength;
        ClearSamples();
        Debug.Log("Animation Started. Length: " + currentSampleLength);

        //reset timers
        List<string> keys = new List<string>();
        foreach (KeyValuePair<string, int> pair in sampleTimers)
        {
            keys.Add(pair.Key);
        }

        foreach (string key in keys)
        {
            sampleTimers[key] = 1; //sample first frame
        }
            
        
    }

    void ProcessPoseStart(string[] info)
    {
        string skeletonName = info[1];
        //Do sampling
        if (sampleRate != 0)
        {
            sampleTimers[skeletonName] -= 1; //decrement sample timer
            if (sampleTimers[skeletonName] == 0) //if timer exceeded
            {
                //sample current state
                frameSamples[skeletonName].Add( DeepCopySkeleton(skeletonName));
                
                //reset timer
                sampleTimers[skeletonName] += sampleRate;
            }
        }
    }

    /// <summary>
    /// Clears all stored frame samples
    /// </summary>
    void ClearSamples()
    {
        foreach (KeyValuePair<string, List<Transform>> skeletonSamples in frameSamples)
        {
            string skeletonName = skeletonSamples.Key;
            List<Transform> samples = skeletonSamples.Value;

            foreach(Transform sampleRoot in samples)
            {
                Destroy(sampleRoot.gameObject);
            }

            samples.Clear();
        }
    }

    Transform DeepCopySkeleton(string skeletonName)
    {
        Transform originalBase = skeletons[skeletonName]["Hips"];
        Transform newBase = Instantiate(originalBase, originalBase.parent);
        Color origColor = originalBase.GetComponent<DrawLinkToParent>().drawColor;

        int numSamples = frameSamples[skeletonName].Count;

        Color sampleColor = Color.Lerp(sampleTint, origColor, sampleColorStep * numSamples);

        foreach (DrawLinkToParent component in newBase.GetComponentsInChildren<DrawLinkToParent>())
            component.drawColor = sampleColor;
        
        return newBase;
    }

    void OnGUI() {
        if(_predictionTimes.Count == _maxPredictionTimeCount)
        {
            decimal _predictionTime = 0;
            foreach(float t in _predictionTimes)
            {
                _predictionTime += (decimal)t;
            }
            
            _predictionTime /= _predictionTimes.Count;
            int fps = (int)(1.0m/_predictionTime);
            GUILayout.TextField(fps.ToString());
            _predictionTime = decimal.Round(_predictionTime, 5);
            GUILayout.TextField(_predictionTime.ToString());
            
        }
        decimal min = decimal.Round((decimal)_minPredictionTime, 5);
        decimal max = decimal.Round((decimal)_maxPredictionTime, 5);
        GUILayout.TextField("Min prediction time: " + min.ToString());
        GUILayout.TextField("Max prediction time: " + max.ToString());
    }
}
