using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DrawLinkToParent : MonoBehaviour
{
    public Color originalColor = Color.blue;
    public Color predictionColor = Color.red;

    public Color drawColor;
    

    // Update is called once per frame
    void Update()
    {
        if (transform.parent == null || transform.parent.GetComponent<DrawLinkToParent>() == null) return;
        
        if(transform.parent != null && !transform.parent.name.Equals("Hips"))
        {
            Debug.DrawLine(transform.position, transform.parent.position, drawColor);
        }    
    }

    private void OnDrawGizmos()
    {
        if (transform.parent == null || transform.parent.GetComponent<DrawLinkToParent>() == null) return;
        
        if(transform.parent != null && !transform.parent.name.Equals("Hips"))
        {
            Debug.DrawLine(transform.position, transform.parent.position, drawColor);
        }    
    }
}
